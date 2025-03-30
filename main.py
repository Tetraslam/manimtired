import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from manim import Line, VGroup
from pydantic import BaseModel, Field

# Parse CLI args for initial LLM selection
parser = argparse.ArgumentParser(description="ManimTired - AI-powered Manim animations")
parser.add_argument("--llm", choices=["cerebras", "gpt4o", "o3"], default="cerebras", help="LLM to use for code generation")
args = parser.parse_args()

# Load environment variables (for API URLs/keys)
load_dotenv()
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not CEREBRAS_BASE_URL and args.llm == "cerebras":
    raise ValueError("CEREBRAS_BASE_URL not found in .env")
if not OPENAI_BASE_URL and args.llm in ["gpt4o", "o3"]:
    raise ValueError("OPENAI_BASE_URL not found in .env")

# Helper: Create LLM client based on selected model
def create_llm(model_key: str, for_planning: bool = False):
    """Initialize an LLM client based on model key (for code or planning)."""
    max_tokens = 4000 if for_planning else 5000  # Use higher limit for planning if needed
    if model_key == "cerebras":
        if not CEREBRAS_BASE_URL:
            raise ValueError("CEREBRAS_BASE_URL not configured.")
        return ChatCerebras(
            model_name="llama-3.3-70b",
            base_url=CEREBRAS_BASE_URL,
            temperature=0.7,
            max_tokens=max_tokens,
        )
    elif model_key == "o3":
        if not OPENAI_BASE_URL:
            raise ValueError("OPENAI_BASE_URL not configured.")
        return ChatOpenAI(
            model="o3-mini",
            base_url=OPENAI_BASE_URL,
            max_tokens=max_tokens,
        )
    elif model_key == "gpt4o":
        if not OPENAI_BASE_URL:
            raise ValueError("OPENAI_BASE_URL not configured.")
        return ChatOpenAI(
            model="gpt-4o",
            base_url=OPENAI_BASE_URL,
            temperature=0.7,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM selection: {model_key}")

# Define JSON output schema for Manim code generation
class ManimCode(BaseModel):
    """Schema for Manim code generation output."""
    code: str = Field(description="The complete Python code for the Manim scene")

parser_obj = JsonOutputParser(pydantic_object=ManimCode)

# Set up prompt template for code generation chain with complex animation limits removed
code_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Manim code generator specialized in creating educational content. Your task is to generate Manim code that effectively communicates complex concepts through animations. You MUST output ONLY valid JSON matching this schema:
{format_instructions}

Your code MUST follow these educational content guidelines:

1. EDUCATIONAL CONTENT:
   - Include the ACTUAL educational content provided in the prompt
   - Break down complex explanations into digestible parts
   - Present information in a logical sequence
   - Use visual elements that directly support the textual explanations

2. CODE STRUCTURE:
   - Define a Manim scene class named `GeneratedScene` that inherits from `Scene`
   - Format the animation as a slideshow with distinct sections
   - Include proper imports (from manim import * \n from manim import FadeIn, AnimationGroup, UpdateFromAlphaFunc) and do not use the from manim import * on the same line as the subsequent manim imports
   - Use proper Python syntax and indentation

3. VISUAL LAYOUT:
   - Place explanatory text on the LEFT side of the screen
   - Place diagrams/visuals on the RIGHT side
   - Maintain appropriate spacing between elements
   - Use appropriate font sizes (titles larger than explanations)

4. ANIMATION TECHNIQUES:
   - Animate text appearance using Write() or AddTextLetterByLetter()
   - Animate diagrams using Create() or other suitable animations
   - Include self.wait() calls for pacing (e.g., after each slide)
"""),
    ("human", "{prompt}")
])

# Initialize default LLM and chain for code generation (will be updated per GUI selection)
llm = create_llm(args.llm, for_planning=False)
chain = code_prompt | llm | parser_obj

## Utility Functions for Code Fixes and Checks

def fix_syntax_errors(code: str) -> str:
    """Fix common syntax issues in the generated code (unclosed parentheses, etc.)."""
    lines = code.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        fixed_lines.append(line)
        # Ensure self.play( calls have matching ')'
        if "self.play(" in line and not line.strip().endswith(")") and line.count("(") > line.count(")"):
            j = i + 1
            found_closing = False
            while j < len(lines) and not found_closing:
                next_line = lines[j].strip()
                if next_line.endswith(")"):
                    found_closing = True
                elif not next_line:
                    # If an empty line, assume closing parenthesis was omitted
                    fixed_lines[-1] += ")"
                    found_closing = True
                j += 1
            if not found_closing:
                fixed_lines[-1] += ")"
        i += 1
    return '\n'.join(fixed_lines)

def fix_common_manim_issues(code_content: str) -> str:
    """Patch common Manim issues (NumberPlane, large shapes, etc.) for safer rendering."""
    fixed_code = code_content
    # Replace NumberPlane with simplified grid (SimpleGrid)
    if "NumberPlane" in fixed_code:
        fixed_code = fixed_code.replace("NumberPlane", "SimpleGrid")
        if "class SimpleGrid" not in fixed_code:
            grid_class = """
# Simplified grid to avoid NumberPlane issues
class SimpleGrid(VGroup):
    def __init__(self, x_range=(-8, 8, 1), y_range=(-5, 5, 1), **kwargs):
        super().__init__(**kwargs)
        for x in range(int(x_range[0]), int(x_range[1]) + 1, int(x_range[2])):
            self.add(Line(np.array([x, y_range[0], 0]), np.array([x, y_range[1], 0]), color=BLUE_D, stroke_width=0.5))
        for y in range(int(y_range[0]), int(y_range[1]) + 1, int(y_range[2])):
            self.add(Line(np.array([x_range[0], y, 0]), np.array([x_range[1], y, 0]), color=BLUE_D, stroke_width=0.5))
"""
            # Insert SimpleGrid class definition before GeneratedScene
            insert_at = fixed_code.find("class GeneratedScene")
            fixed_code = (fixed_code[:insert_at] + grid_class + fixed_code[insert_at:]) if insert_at != -1 else fixed_code + grid_class
    # Ensure parentheses in self.play calls are closed
    lines = fixed_code.split('\n')
    for idx, line in enumerate(lines):
        if 'self.play(' in line:
            open_paren = line.count('(')
            close_paren = line.count(')')
            if open_paren > close_paren:
                lines[idx] += ')' * (open_paren - close_paren)
    fixed_code = '\n'.join(lines)
    # Introduce safe_create for Circle/Arrow if Create is used on them
    if ("Create(Circle" in fixed_code or "Create(Arrow" in fixed_code) and "def safe_create" not in fixed_code:
        safe_create_func = """
# Safe creation for large shapes (prevents hang-ups)
def safe_create(mobject):
    original_opacity = mobject.get_opacity()
    mobject.set_opacity(0)
    return AnimationGroup(
        FadeIn(mobject),
        UpdateFromAlphaFunc(mobject, lambda m, a: m.set_opacity(original_opacity * a))
    )
"""
        insert_at = fixed_code.find("class GeneratedScene")
        fixed_code = (fixed_code[:insert_at] + safe_create_func + fixed_code[insert_at:]) if insert_at != -1 else fixed_code + safe_create_func
    fixed_code = fixed_code.replace("Create(Circle", "safe_create(Circle").replace("Create(Arrow", "safe_create(Arrow")
    # Clamp Circle radius to [0.5, 3.0]
    fixed_code = re.sub(r'Circle\(radius=([0-9.]+)', lambda m: f'Circle(radius={min(max(float(m.group(1)), 0.5), 3.0)}', fixed_code)
    # Clamp Arrow length to <= 3.0 (for vectors like LEFT*10)
    fixed_code = re.sub(r'Arrow\(.*?(LEFT|RIGHT|UP|DOWN)\s*\*\s*([0-9.]+)', lambda m: f'Arrow({m.group(1)}*{min(float(m.group(2)), 3.0)}', fixed_code)
    # Ensure safe_create imports
    if "safe_create" in fixed_code and "AnimationGroup" not in fixed_code:
        fixed_code = fixed_code.replace("from manim import *", "from manim import *\nfrom manim import AnimationGroup, UpdateFromAlphaFunc")
    return fixed_code

# Modified: Remove all fallbacks that limit complex animations by always returning no problems.
def check_manim_code_for_problems(code: str) -> Tuple[bool, Optional[str]]:
    """Detect problematic elements (limiting checks removed to allow complex animations)."""
    return False, None

def generate_manim_code(user_prompt: str) -> str:
    """Generate Manim code from user prompt using the LLM chain and fix issues."""
    try:
        print(f"Generating code from prompt: {user_prompt[:50]}...")
        response = chain.invoke({"format_instructions": parser_obj.get_format_instructions(), "prompt": user_prompt})
        code = response["code"]
        # Basic validation of output
        if "class GeneratedScene" not in code or "Scene" not in code:
            raise ValueError("Generated code is missing a Scene class.")
        if "from manim import" not in code:
            code = "from manim import *\n\n" + code
        # Fix and check for issues
        problematic, reason = check_manim_code_for_problems(code)
        if problematic:
            print(f"Warning: {reason} – applying fixes...")
        code = fix_syntax_errors(code)
        code = fix_common_manim_issues(code)
        problematic, reason = check_manim_code_for_problems(code)
        if problematic:
            raise ValueError(f"Final code still problematic: {reason}")
        code = code.encode('utf-8').decode('utf-8')  # Ensure proper encoding
        print("Manim code generation successful.")
        return code
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('"', "'")
        if "Invalid json output" in error_msg:
            error_msg = "LLM did not produce valid code (JSON parse failed). Try a simpler prompt."
        else:
            error_msg = f"Code generation error: {error_msg}"
        print(error_msg)
        raise ValueError(error_msg)

def fix_script_syntax_errors(script_content: str, error_msg: str) -> str:
    """Attempt to auto-fix any syntax errors in the script via the LLM."""
    try:
        fixed_script = script_content
        # Quick fix for common unterminated string (missing quote)
        if "unterminated string literal" in error_msg:
            match = re.search(r'line (\d+)', error_msg)
            if match:
                line_no = int(match.group(1))
                lines = fixed_script.splitlines()
                if 0 <= line_no - 1 < len(lines):
                    line_text = lines[line_no - 1]
                    if line_text.count("'") % 2 == 1:
                        lines[line_no - 1] += "'"  # add missing single quote
                    elif line_text.count('"') % 2 == 1:
                        lines[line_no - 1] += '"'
                    fixed_script = "\n".join(lines)
                    print(f"Added missing quote at line {line_no}.")
        # Check if quick fix resolved issue
        try:
            compile(fixed_script, "<string>", "exec")
            print("Script compiles after quick fix.")
            return fixed_script
        except SyntaxError:
            pass  # If still failing, proceed to LLM-based fix
        # Use LLM to fix syntax errors without altering content
        fix_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a Manim code debugging expert. Fix the syntax errors in the script below. Error details: {error_msg}. Only return corrected code."),
            ("human", "{script}")
        ])
        fix_response = llm.invoke(fix_prompt.format_messages(script=script_content))
        fixed_script = fix_response.content.strip()
        try:
            compile(fixed_script, "<string>", "exec")
            print("Script compiles after LLM fix.")
            return fixed_script
        except SyntaxError as new_err:
            print(f"LLM fix attempt failed: {new_err}")
            return script_content  # Return original if fix unsuccessful
    except Exception as e:
        print(f"Error in fix_script_syntax_errors: {e}")
        return script_content

## Agent Classes

class ExplanationAgent:
    """Agent to generate detailed explanations for each slide title."""
    def execute(self, outline: List[str], prompt: str, llm_client) -> Dict[str, str]:
        try:
            system_msg = ("You are an expert educator. Given the topic and slide titles, write a detailed explanation (at least 3 sentences) for each slide title.")
            user_msg = "Topic: " + prompt + "\nSlide Titles:\n" + "\n".join([f"{i}. {title}" for i, title in enumerate(outline, 1)])
            conversation = ChatPromptTemplate.from_messages([("system", system_msg), ("human", user_msg)])
            response = llm_client.invoke(conversation.format_messages())
            content = response.content.strip()
            explanations = {}
            # Parse output for "Slide X" or "EXPLANATION SLIDE X" markers
            for i in range(1, 5):
                regex = re.compile(rf"(?i)(?:Slide {i}|EXPLANATION SLIDE {i})[:\s]*(.*)")
                match = regex.search(content)
                if match:
                    explanations[str(i)] = match.group(1).strip()
            return explanations
        except Exception as e:
            print(f"ExplanationAgent error: {e}")
            return {}

class DiagramAgent:
    """Agent to generate visual diagram descriptions for each slide."""
    def execute(self, outline: List[str], prompt: str, explanations: Dict[str, str], llm_client) -> Dict[str, str]:
        try:
            system_msg = ("You are a visual designer. For each slide's title and explanation, suggest a concise diagram description (1-2 sentences) to illustrate that slide.")
            # Prepare the user message with each slide's content
            user_lines = []
            for i, title in enumerate(outline, 1):
                exp_text = explanations.get(str(i), "")
                user_lines.append(f"{i}. {title}\nExplanation: {exp_text}")
            user_msg = "Topic: " + prompt + "\n" + "\n".join(user_lines)
            conversation = ChatPromptTemplate.from_messages([("system", system_msg), ("human", user_msg)])
            response = llm_client.invoke(conversation.format_messages())
            content = response.content.strip()
            diagrams = {}
            for i in range(1, 5):
                regex = re.compile(rf"(?i)(?:Slide {i}|DIAGRAM SLIDE {i})[:\s]*(.*)")
                match = regex.search(content)
                if match:
                    diagrams[str(i)] = match.group(1).strip()
            return diagrams
        except Exception as e:
            print(f"DiagramAgent error: {e}")
            return {}

class ScriptAgent:
    """Agent to generate the final Manim script from the content plan."""
    def __init__(self):
        self.fallback_used = False

    def execute(self, plan: dict, prompt: str) -> str:
        # Build a detailed code-generation prompt using plan content
        prompt_text = "Create a Manim animation for an educational slideshow with alternating text (explanation) and visual (diagram) slides:\n\n"
        for i in range(1, 5):
            idx = str(i)
            title = plan["explanation_slides"][idx]["title"]
            explanation = plan["explanation_slides"][idx]["explanation"]
            diagram = plan["drawing_slides"][idx]["diagram"]
            prompt_text += f"EXPLANATION SLIDE {i}:\n- TITLE: {title}\n- EXPLANATION: {explanation}\n\n"
            prompt_text += f"DRAWING SLIDE {i}:\n- DIAGRAM: {diagram}\n\n"
        prompt_text += """Follow these code guidelines strictly:

- Use `from manim import *` and define class GeneratedScene(Scene) with construct().
- Alternate between text slides and diagram slides.
- **Explanation slide**: show Title (top center) and explanation text below (centered), use Write() for text and self.wait().
- **Drawing slide**: illustrate concept with simple shapes (Circle, Square, Arrow, etc.), minimal text labels. Use Create() or FadeIn animations and self.wait().
- Always clear scene (self.clear()) when moving to the next slide.
"""
        try:
            code = generate_manim_code(prompt_text)
            return code
        except Exception as e:
            # On failure, fallback to a basic script (ensures stability)
            print(f"ScriptAgent failed to generate code: {e}. Using fallback script.")
            self.fallback_used = True
            return create_safe_fallback_script(prompt, plan)

def create_safe_fallback_script(prompt: str, plan: dict) -> str:
    """Generate a very simple but guaranteed-to-run Manim script from the plan."""
    script = "from manim import *\n\nclass GeneratedScene(Scene):\n    def construct(self):\n"
    # For each slide, create extremely basic content
    for i in range(1, 5):
        idx = str(i)
        slide_title = plan["explanation_slides"].get(idx, {}).get("title", f"Slide {i}")
        slide_expl = plan["explanation_slides"].get(idx, {}).get("explanation", f"About {prompt}")
        # Break explanation into manageable lines (for text blocks)
        words = slide_expl.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > 60:
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        # Add explanation slide text
        script += f"\n        # Explanation Slide {i}: {slide_title}\n"
        script += f"        title_{i} = Text(\"{slide_title}\", font_size=40).to_edge(UP)\n"
        for j, text_line in enumerate(lines):
            line_tag = f"text_{i}_{j}"
            if j == 0:
                script += f"        {line_tag} = Text(\"{text_line}\", font_size=30).next_to(title_{i}, DOWN, buff=0.5)\n"
            else:
                script += f"        {line_tag} = Text(\"{text_line}\", font_size=30).next_to(text_{i}_{j-1}, DOWN, buff=0.3)\n"
        script += f"        self.play(Write(title_{i}))\n        self.wait(0.5)\n"
        for j in range(len(lines)):
            script += f"        self.play(Write(text_{i}_{j}))\n        self.wait(0.5)\n"
        script += "        self.wait(1)\n        self.clear()\n"
        # Add diagram slide with a single shape labeled by title
        script += f"\n        # Drawing Slide {i}: Visualization for {slide_title}\n"
        desc = plan["drawing_slides"].get(idx, {}).get("diagram", "")
        shape_tag = f"shape_{i}"
        label_tag = f"label_{i}"
        # Decide shape type by keywords in diagram description
        if any(word in desc.lower() for word in ["square", "rectangle"]):
            script += f"        {shape_tag} = Square(side_length=3, color=GREEN)\n"
        elif "triangle" in desc.lower():
            script += f"        {shape_tag} = Triangle(color=YELLOW).scale(2)\n"
        elif "arrow" in desc.lower() or "flow" in desc.lower():
            script += f"        {shape_tag} = Arrow(start=LEFT*2, end=RIGHT*2, color=RED)\n"
        else:
            script += f"        {shape_tag} = Circle(radius=2, color=BLUE)\n"
        script += f"        {label_tag} = Text(\"{slide_title}\", font_size=24).next_to({shape_tag}, DOWN)\n"
        script += f"        self.play(Create({shape_tag}))\n        self.wait(0.5)\n"
        script += f"        self.play(Write({label_tag}))\n        self.wait(1)\n        self.clear()\n"
    script += "        # End Slide\n        end_text = Text(\"Thank you for watching!\", font_size=48)\n"
    script += "        self.play(Write(end_text))\n        self.wait(2)\n"
    return script

## Generation Thread and Rendering

def generation_thread(prompt: str):
    """Background thread that orchestrates the generation process (planning -> script -> rendering)."""
    global planning_agent, llm, chain
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = os.path.join("generated_videos", f"script_{timestamp}.py")
        os.makedirs("generated_videos", exist_ok=True)
        os.makedirs("temp_scripts", exist_ok=True)
        # Update UI: starting content planning
        dpg.configure_item("status_text", default_value="Planning content outline...", color=[255, 255, 0])
        selected_llm = dpg.get_value("llm_selector")
        # Validate that required API config exists for selected LLM
        if selected_llm == "cerebras" and not CEREBRAS_BASE_URL:
            dpg.configure_item("status_text", default_value="Cerebras API not configured.", color=[255, 0, 0])
            dpg.configure_item("error_text", default_value="CerebrAS API base URL missing. Check your .env configuration.")
            dpg.configure_item("error_group", show=True)
            return _reset_ui_after_failure()
        if selected_llm in ["gpt4o", "o3"] and not OPENAI_BASE_URL:
            dpg.configure_item("status_text", default_value="OpenAI API not configured.", color=[255, 0, 0])
            dpg.configure_item("error_text", default_value="OpenAI API base URL missing. Check your .env configuration.")
            dpg.configure_item("error_group", show=True)
            return _reset_ui_after_failure()
        # Reinitialize planning agent and code LLM with selected model
        planning_agent = PlanningAgent(selected_llm)
        llm = create_llm(selected_llm, for_planning=False)
        chain = code_prompt | llm | parser_obj
        # Generate slide titles (outline)
        outline = planning_agent.generate_outline(prompt)
        if not outline or len(outline) < 4:
            dpg.configure_item("status_text", default_value="Failed to plan content.", color=[255, 0, 0])
            dpg.configure_item("error_text", default_value="Could not create content outline. Try a different prompt.")
            dpg.configure_item("error_group", show=True)
            return _reset_ui_after_failure()
        # Generate explanations and diagrams
        dpg.configure_item("status_text", default_value="Generating explanation slides...", color=[255, 255, 0])
        explanations = planning_agent.agents["explanation"].execute(outline, prompt, planning_agent.planning_llm)
        dpg.configure_item("status_text", default_value="Generating diagram slides...", color=[255, 255, 0])
        diagrams = planning_agent.agents["diagram"].execute(outline, prompt, explanations, planning_agent.planning_llm)
        # Construct the final content plan (ensuring each slide has content)
        plan = {"explanation_slides": {}, "drawing_slides": {}}
        try:
            for i in range(1, 5):
                idx = str(i)
                title = outline[i-1] if i-1 < len(outline) else f"{prompt} (Slide {i})"
                
                # Get explanation or use LLM to generate one if missing
                expl_text = explanations.get(idx, "")
                if not expl_text or len(expl_text.split()) < 8:
                    try:
                        # Try to get from fallback plan if we have one
                        expl_text = planning_agent._create_fallback_explanation_slide(i, prompt)["explanation"]
                    except:
                        # If that fails too, make a direct LLM call
                        system_prompt = f"Write a detailed explanation (3+ sentences) for a slide titled '{title}' about {prompt}."
                        expl_response = planning_agent.planning_llm.invoke(system_prompt)
                        expl_text = expl_response.content.strip()
                        
                plan["explanation_slides"][idx] = {"title": title, "explanation": expl_text}
                
                # Get diagram description or use LLM to generate one if missing
                diag_text = diagrams.get(idx, "")
                if not diag_text or len(diag_text.split()) < 3:
                    try:
                        # Try to get from fallback plan if we have one
                        diag_text = planning_agent._create_fallback_drawing_slide(i, prompt)["diagram"]
                    except:
                        # If that fails too, make a direct LLM call
                        system_prompt = f"Describe a visual diagram (1-2 sentences) that would illustrate a slide titled '{title}' about {prompt}."
                        diag_response = planning_agent.planning_llm.invoke(system_prompt)
                        diag_text = diag_response.content.strip()
                        
                plan["drawing_slides"][idx] = {"diagram": diag_text}
        except Exception as e:
            print(f"Error building content plan: {e}. Generating complete fallback plan.")
            try:
                # Generate a complete fallback plan if individual slide generation fails
                plan = planning_agent._create_fallback_plan(prompt)
            except Exception as fallback_error:
                print(f"Fallback plan generation failed: {fallback_error}")
                dpg.configure_item("status_text", default_value="Failed to generate content.", color=[255, 0, 0])
                dpg.configure_item("error_text", default_value=f"Error creating content plan: {fallback_error}")
                dpg.configure_item("error_group", show=True)
                return _reset_ui_after_failure()
        # Update UI: starting code generation
        dpg.configure_item("status_text", default_value="Generating Manim script...", color=[255, 255, 0])
        script_agent = planning_agent.agents["script"]
        script_content = script_agent.execute(plan, prompt)
        # Show warning if fallback script was used (due to complexity issues)
        if getattr(script_agent, "fallback_used", False):
            dpg.configure_item("warning_text", default_value="The script was simplified due to complexity.", color=[255, 165, 0])
            dpg.configure_item("warning_group", show=True)
        script_content = fix_common_manim_issues(script_content)  # Final safety pass
        # Save script to file for rendering
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        # Validate script syntax before rendering
        try:
            compile(script_content, script_path, 'exec')
            print("Script validation passed (no syntax errors).")
        except SyntaxError as e:
            line_no = e.lineno or 0
            err_msg = f"Syntax error at line {line_no}: {e.msg}"
            if line_no:
                line_text = script_content.splitlines()[line_no - 1] if line_no <= len(script_content.splitlines()) else ""
                err_msg += f" (line content: {line_text})"
            print(f"Syntax error detected: {err_msg}")
            dpg.configure_item("status_text", default_value="Fixing syntax errors...", color=[255, 165, 0])
            fixed_script = fix_script_syntax_errors(script_content, err_msg)
            if fixed_script != script_content:
                script_content = fixed_script
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                print("Syntax errors fixed automatically.")
            else:
                dpg.configure_item("status_text", default_value="Failed to fix syntax errors.", color=[255, 0, 0])
                dpg.configure_item("error_text", default_value=err_msg)
                dpg.configure_item("error_group", show=True)
                return _reset_ui_after_failure()
        # Update UI: start rendering the Manim video
        dpg.configure_item("status_text", default_value="Rendering animation...", color=[255, 255, 0])
        def update_render_status(msg):
            dpg.configure_item("status_text", default_value=msg, color=[255, 255, 0])
            if dpg.get_value("cancel_render"):
                # Trigger cancellation if user clicked "Cancel Render"
                raise Exception("Rendering canceled by user")
        video_path, success = run_manim_render(script_path, "generated_videos", status_callback=update_render_status, timeout_seconds=180)
        if not video_path or not success:
            error_note = "Rendering failed or timed out"
            if dpg.get_value("cancel_render"):
                error_note = "Rendering was canceled by user"
            dpg.configure_item("status_text", default_value=error_note, color=[255, 0, 0])
            dpg.configure_item("error_text", default_value=error_note)
            dpg.configure_item("error_group", show=True)
        else:
            thumb_path = os.path.join("generated_videos", f"thumb_{timestamp}.png")
            generate_thumbnail(video_path, thumb_path)
            dpg.configure_item("status_text", default_value="Animation generated successfully!", color=[0, 255, 0])
            dpg.configure_item("video_path_text", default_value=f"Video path: {video_path}")
            # Display result section with video/thumbnail
            dpg.configure_item("result_group", show=True)
            if os.path.exists(thumb_path):
                try:
                    width, height, channels, data = dpg.load_image(thumb_path)
                    with dpg.texture_registry():
                        dpg.add_static_texture(width=width, height=height, default_value=data, tag="thumbnail")
                    dpg.add_image(texture_tag="thumbnail", tag="thumbnail_img", parent="result_group", width=width, height=height)  # Ensure it shows in result group
                except Exception as e:
                    print(f"Thumbnail load error: {e}")
            print(f"Video created at {video_path}")
    except Exception as e:
        err = str(e).replace('\n', ' ').replace('"', "'")
        print(f"Error in generation thread: {err}")
        if "canceled by user" in err:
            dpg.configure_item("status_text", default_value="Rendering canceled", color=[255, 165, 0])
        else:
            dpg.configure_item("status_text", default_value=f"Error: {err}", color=[255, 0, 0])
            dpg.configure_item("error_text", default_value=f"Error: {err}")
            dpg.configure_item("error_group", show=True)
    finally:
        # Re-enable UI controls regardless of success or failure
        dpg.configure_item("generate_btn", enabled=True)
        dpg.configure_item("prompt_input", enabled=True)
        dpg.configure_item("llm_selector", enabled=True)
        dpg.configure_item("loading_indicator", show=False)
        dpg.configure_item("cancel_btn", show=False)

# Helper to reset UI in case of failure before rendering completes
def _reset_ui_after_failure():
    dpg.configure_item("loading_indicator", show=False)
    dpg.configure_item("cancel_btn", show=False)
    dpg.configure_item("generate_btn", enabled=True)
    dpg.configure_item("prompt_input", enabled=True)
    dpg.configure_item("llm_selector", enabled=True)
    return

def run_manim_render(script_path: str, media_dir: str, status_callback=None, timeout_seconds=180):
    """Execute the Manim rendering process and monitor for timeouts."""
    os.makedirs(media_dir, exist_ok=True)
    # Pre-patch script to ensure safe rendering
    with open(script_path, 'r', encoding='utf-8') as f:
        script_data = f.read()
    script_data = patch_script_for_problematic_objects(script_data)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_data)
    # Command to run Manim (Community v0.19) in silent mode, writing video to media_dir
    cmd = ["manim", "-pqm", script_path, "GeneratedScene", "--media_dir", media_dir]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    video_path = None
    last_animation = -1
    last_anim_time = time.time()
    timed_out = False
    output_log = deque(maxlen=10)
    # Start a timeout watcher thread
    def timeout_watcher():
        nonlocal timed_out
        time.sleep(timeout_seconds)
        if process.poll() is None:  # still running
            timed_out = True
            print(f"Render timeout reached ({timeout_seconds}s). Terminating process.")
            try:
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
            except Exception as e:
                print(f"Error terminating Manim process: {e}")
    threading.Thread(target=timeout_watcher, daemon=True).start()
    # Monitor process output
    try:
        for line in process.stdout:
            if not line and process.poll() is not None:
                break
            output_log.append(line)
            # Parse animation progress lines (e.g., "Animation 1: 0%...")
            if "Animation" in line and "%" in line:
                try:
                    anim_number = int(line.split("Animation ")[1].split(":")[0])
                    progress_match = re.search(r"(\d+)%", line)
                    percent = int(progress_match.group(1)) if progress_match else 0
                    # Check for stuck animations (0% for >30s)
                    if anim_number == last_animation and percent == 0 and time.time() - last_anim_time > 30:
                        print(f"Animation {anim_number} stuck at 0% for too long. Terminating render.")
                        process.terminate()
                        timed_out = True
                        break
                    if anim_number != last_animation:
                        last_animation = anim_number
                        last_anim_time = time.time()
                    # Send status update to UI
                    if status_callback:
                        status_callback(f"Rendering animation {anim_number} ({percent}%)")
                except Exception as e:
                    print(f"Error parsing progress: {e}")
            # Extract final video path from output
            if "File ready at" in line:
                match = re.search(r'File ready at (.*)', line)
                if match:
                    video_path = match.group(1).strip()
            # Log output for debugging
            print(line, end='', flush=True)
    except Exception as e:
        print(f"Exception during rendering: {e}")
        process.terminate()
    process.wait()
    if timed_out:
        print("Manim rendering timed out. Last output lines:")
        for out_line in output_log:
            print(out_line, end='')
        # If video_path not found, try to find a partial video file
        if not video_path:
            videos = glob.glob(os.path.join(media_dir, "videos", "GeneratedScene", "*.mp4"))
            if videos:
                video_path = max(videos, key=os.path.getctime)
                print(f"Partial video found: {video_path}")
    return video_path, process.returncode == 0

def patch_script_for_problematic_objects(script_content: str) -> str:
    """Apply final patches to ensure safe creation of large shapes and clamp sizes."""
    patched = script_content
    # Ensure necessary imports for safe_create
    if "from manim import *" in patched:
        patched = patched.replace("from manim import *", "from manim import *\nfrom manim import FadeIn, AnimationGroup, UpdateFromAlphaFunc")
    # Add safe_create function if needed
    if ("Create(Circle" in patched or "Create(Arrow" in patched) and "def safe_create" not in patched:
        safe_func = """
# Safe creation function for problematic shapes
def safe_create(mobject):
    original_opacity = mobject.get_opacity()
    mobject.set_opacity(0)
    return AnimationGroup(
        FadeIn(mobject),
        UpdateFromAlphaFunc(mobject, lambda m, a: m.set_opacity(original_opacity * a))
    )
"""
        insert_at = patched.find("class GeneratedScene")
        patched = (patched[:insert_at] + safe_func + patched[insert_at:]) if insert_at != -1 else patched + safe_func
    # Replace Create with safe_create for Circles and Arrows
    patched = re.sub(r'Create\((Circle[^)]*)\)', r'safe_create(\1)', patched)
    patched = re.sub(r'Create\((Arrow[^)]*)\)', r'safe_create(\1)', patched)
    # Clamp Circle radius and Arrow length
    patched = re.sub(r'Circle\(radius=([0-9.]+)', lambda m: f'Circle(radius={min(max(float(m.group(1)), 0.5), 3.0)}', patched)
    patched = re.sub(r'Arrow\(.*?(LEFT|RIGHT|UP|DOWN)\s*\*\s*([0-9.]+)', lambda m: f'Arrow({m.group(1)}*{min(float(m.group(2)), 3.0)}', patched)
    # Force simpler arrow tip to avoid defaults that might be complex
    patched = re.sub(r'Arrow\(', r'Arrow(', patched)
    # Look for all Arrow instances and add tip_shape in the correct position
    # Make the pattern more specific to only match Arrow class instantiations
    arrow_pattern = re.compile(r'(?<!\w)Arrow\(([^,]*(?:,[^,]*)*)\)')
    
    def arrow_replacer(match):
        args = match.group(1)
        # If already has tip_shape, don't modify
        if 'tip_shape=' in args:
            return f'Arrow({args})'
        # Count commas to determine how to add the parameter
        comma_count = args.count(',')
        if comma_count >= 1:
            # Has multiple args, add as keyword arg at the end
            return f'Arrow({args}, tip_shape=ArrowTriangleFilledTip)'
        elif args.strip():
            # Has one arg, add comma and keyword
            return f'Arrow({args}, tip_shape=ArrowTriangleFilledTip)'
        else:
            # Empty args
            return f'Arrow(tip_shape=ArrowTriangleFilledTip)'
    
    patched = arrow_pattern.sub(arrow_replacer, patched)
    
    # Also check for any tip_shape added to wait() calls and remove them
    wait_with_tip_pattern = re.compile(r'\.wait\(([^,)]*),\s*tip_shape=ArrowTriangleFilledTip\)')
    patched = wait_with_tip_pattern.sub(r'.wait(\1)', patched)
    
    # Fix the axes.get_graph() color parameter issue
    patched = re.sub(r'(\w+)\s*=\s*axes\.get_graph\((.*?),\s*color=([A-Z_]+)\)', 
                   r'\1 = axes.get_graph(\2)\n        \1.set_stroke(color=\3)', patched)
    
    return patched

def generate_thumbnail(video_path: str, thumb_path: str):
    """Generate a thumbnail (PNG) from the video at 2 seconds (or midpoint if shorter)."""
    try:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
        timestamp = min(2.0, duration / 2)
        (
            ffmpeg.input(video_path, ss=timestamp)
            .filter("scale", 320, -1)
            .output(thumb_path, vframes=1, format="image2", vcodec="png")
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")

def play_video_in_gui(video_path: str):
    """Play rendered video inside the GUI using OpenCV."""
    try:
        dpg.configure_item("status_text", default_value="Playing video in GUI...", color=[0, 255, 0])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            dpg.configure_item("video_path_text", default_value="Error: Cannot open video")
            dpg.configure_item("status_text", default_value="Error: Cannot open video", color=[255, 0, 0])
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Resize the dynamic texture to video dimensions
        dpg.delete_item("video_texture")
        with dpg.texture_registry():
            dpg.add_dynamic_texture(width, height, [0, 0, 0, 255] * width * height, tag="video_texture")
        dpg.configure_item("video_image", texture_tag="video_texture")
        # Thread for video playback
        def video_thread():
            frame_count = 0
            while cap.isOpened() and dpg.get_value("is_playing"):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                    continue
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                dpg.set_value("video_texture", frame_rgba.flatten())
                frame_count += 1
                dpg.set_value("frame_counter", f"Frame: {frame_count}")
                time.sleep(frame_delay)
            cap.release()
            dpg.set_value("is_playing", False)
            dpg.configure_item("play_in_gui_btn", label="Play in GUI")
            dpg.configure_item("status_text", default_value="Video playback finished", color=[0, 255, 0])
        dpg.set_value("is_playing", True)
        dpg.configure_item("play_in_gui_btn", label="Stop")
        threading.Thread(target=video_thread, daemon=True).start()
    except Exception as e:
        dpg.configure_item("video_path_text", default_value=f"Error playing video: {e}")
        dpg.configure_item("status_text", default_value=f"Error playing video: {e}", color=[255, 0, 0])

## PlanningAgent orchestrates the high-level planning and holds agents

class PlanningAgent:
    """Planning agent for creating the content outline and coordinating specialized agents."""
    def __init__(self, llm_choice: Optional[str] = None):
        model_key = llm_choice if llm_choice else args.llm
        self.agents = {
            "explanation": ExplanationAgent(),
            "diagram": DiagramAgent(),
            "script": ScriptAgent()
        }
        # Use a separate LLM client for content planning (outline, explanations, diagrams)
        self.planning_llm = create_llm(model_key, for_planning=True)

    def generate_outline(self, prompt: str) -> List[str]:
        """Generate 4 slide titles as an outline for the prompt."""
        try:
            system_msg = "You are an expert educational planner. Provide 4 clear, concise slide titles to explain the topic."
            user_msg = f"Topic: {prompt}\nList 4 slide titles:"
            conversation = ChatPromptTemplate.from_messages([("system", system_msg), ("human", user_msg)])
            response = self.planning_llm.invoke(conversation.format_messages())
            content = response.content.strip()
            titles = []
            for line in content.splitlines():
                # Accept formats like "1. Title", "Slide 1: Title", or just "Title"
                cleaned = re.sub(r'^\d+[\).\s:-]*', '', line).strip()
                if cleaned:
                    titles.append(cleaned)
            
            # If we don't have enough titles, use the full fallback plan generation
            if len(titles) < 4:
                print(f"Outline generation returned only {len(titles)} titles, using fallback LLM planning.")
                fallback = self._create_fallback_plan(prompt)
                titles = [fallback["explanation_slides"][str(i)]["title"] for i in range(1, 5)]
                    
            return titles[:4]
        except Exception as e:
            print(f"Outline generation error: {e}")
            # Try fallback plan as a last resort
            try:
                fallback = self._create_fallback_plan(prompt)
                return [fallback["explanation_slides"][str(i)]["title"] for i in range(1, 5)]
            except Exception as fallback_error:
                print(f"Fallback plan generation also failed: {fallback_error}")
                raise ValueError(f"Failed to generate outline: {e}")

    def _create_fallback_plan(self, prompt: str) -> dict:
        """Create a comprehensive LLM-generated content plan when step-by-step generation fails."""
        try:
            system_msg = """You are an expert educational content planner. Create a complete 4-slide plan for a Manim animation based on the given topic.
For each of the 4 slides, provide:
1. A clear, concise title
2. A detailed explanation (3+ sentences)
3. A visual diagram description (what should be visualized)

Output as a structured, well-formatted response with clear slide-by-slide organization.
"""
            user_msg = f"Topic: {prompt}\nPlease create a comprehensive 4-slide educational plan:"
            conversation = ChatPromptTemplate.from_messages([("system", system_msg), ("human", user_msg)])
            response = self.planning_llm.invoke(conversation.format_messages())
            content = response.content.strip()
            
            # Initialize the plan structure
            plan = {"explanation_slides": {}, "drawing_slides": {}}
            
            # Extract slide content using regex
            slides_content = re.split(r'(?i)slide\s*\d+[:.]\s*|(?<=\n)(?=\d+\.)', content)
            current_slide = 1
            
            # If we couldn't split properly, try to parse the whole response
            if len(slides_content) <= 1:
                # Extract titles with regex
                titles = re.findall(r'(?i)(?:slide\s*(\d+)|^\s*(\d+))[:.]\s*([^\n]+)', content, re.MULTILINE)
                explanations = re.findall(r'(?i)explanation[:\s]*([^\n].*?)(?=\n\s*(?:diagram|visual|slide|$))', content, re.DOTALL)
                diagrams = re.findall(r'(?i)(?:diagram|visual)[:\s]*([^\n].*?)(?=\n\s*(?:slide|$))', content, re.DOTALL)
                
                # Create plan from extracted content
                for i in range(1, 5):
                    idx = str(i)
                    # Find title or create generic one
                    title = f"Slide {i}: {prompt}"
                    for t in titles:
                        if t[0] == str(i) or t[1] == str(i):
                            title = t[2].strip()
                            break
                    
                    # Get explanation if available
                    explanation = f"This slide covers important aspects of {prompt}."
                    if i-1 < len(explanations):
                        explanation = explanations[i-1].strip()
                    
                    # Get diagram if available
                    diagram = f"A visual representation of {prompt}."
                    if i-1 < len(diagrams):
                        diagram = diagrams[i-1].strip()
                    
                    plan["explanation_slides"][idx] = {"title": title, "explanation": explanation}
                    plan["drawing_slides"][idx] = {"diagram": diagram}
            else:
                # Process each slide section
                for section in slides_content:
                    if not section.strip():
                        continue
                    
                    if current_slide > 4:
                        break
                        
                    idx = str(current_slide)
                    
                    # Extract title (first line or "Title: X" pattern)
                    title_match = re.search(r'(?i)(?:^|\n)(?:title[:\s]*)?([^\n]+)', section)
                    title = title_match.group(1).strip() if title_match else f"Slide {current_slide}: {prompt}"
                    
                    # Extract explanation
                    explanation_match = re.search(r'(?i)explanation[:\s]*([^\n].*?)(?=\n\s*(?:diagram|visual|$))', section, re.DOTALL)
                    explanation = explanation_match.group(1).strip() if explanation_match else f"This slide discusses {title.lower()}."
                    
                    # Extract diagram description
                    diagram_match = re.search(r'(?i)(?:diagram|visual)[:\s]*([^\n].*)', section, re.DOTALL)
                    diagram = diagram_match.group(1).strip() if diagram_match else f"A visual representation of {title.lower()}."
                    
                    plan["explanation_slides"][idx] = {"title": title, "explanation": explanation}
                    plan["drawing_slides"][idx] = {"diagram": diagram}
                    current_slide += 1
            
            # Ensure we have 4 slides
            for i in range(1, 5):
                idx = str(i)
                if idx not in plan["explanation_slides"]:
                    # Make a second attempt with a direct title request
                    system_msg = f"Create a title, explanation paragraph, and diagram description for a slide about '{prompt}'"
                    retried_response = self.planning_llm.invoke(system_msg)
                    content = retried_response.content.strip()
                    
                    title_match = re.search(r'(?i)(?:^|\n)(?:title[:\s]*)?([^\n]+)', content)
                    title = title_match.group(1).strip() if title_match else f"Slide {i}: {prompt}"
                    
                    explanation_match = re.search(r'(?i)explanation[:\s]*([^\n].*?)(?=\n\s*(?:diagram|visual|$))', content, re.DOTALL)
                    explanation = explanation_match.group(1).strip() if explanation_match else f"This slide discusses {prompt}."
                    
                    diagram_match = re.search(r'(?i)(?:diagram|visual)[:\s]*([^\n].*)', content, re.DOTALL)
                    diagram = diagram_match.group(1).strip() if diagram_match else f"A visual representation of {prompt}."
                    
                    plan["explanation_slides"][idx] = {"title": title, "explanation": explanation}
                    plan["drawing_slides"][idx] = {"diagram": diagram}
            
            print("Successfully created a comprehensive content plan via LLM.")
            return plan
            
        except Exception as e:
            print(f"LLM-based fallback plan generation failed: {e}")
            raise ValueError(f"Failed to generate any content plan for '{prompt}'. Error: {e}")

    def _create_fallback_explanation_slide(self, slide_num: int, prompt: str) -> dict:
        """Fallback explanation for an individual slide."""
        return self._create_fallback_plan(prompt)["explanation_slides"][str(slide_num)]

    def _create_fallback_drawing_slide(self, slide_num: int, prompt: str) -> dict:
        """Fallback diagram description for an individual slide."""
        return self._create_fallback_plan(prompt)["drawing_slides"][str(slide_num)]

## GUI Setup with DearPyGUI

def main():
    dpg.create_context()
    dpg.create_viewport(title="ManimTired", width=900, height=700)
    global planning_agent
    planning_agent = PlanningAgent()  # Initialize with default args.llm

    # Shared state values
    with dpg.value_registry():
        dpg.add_bool_value(tag="is_playing", default_value=False)
        dpg.add_string_value(tag="frame_counter", default_value="Frame: 0")
        dpg.add_bool_value(tag="cancel_render", default_value=False)
    # Texture registry for video playback
    with dpg.texture_registry(tag="texture_registry"):
        dpg.add_static_texture(1, 1, [0, 0, 0, 255], tag="blank_texture")
        dpg.add_dynamic_texture(800, 450, [0, 0, 0, 255] * 800 * 450, tag="video_texture")
    # Main application window
    with dpg.window(label="ManimTired - AI-Powered Manim Animations", tag="primary", width=900, height=700):
        # Header
        with dpg.group(horizontal=True):
            dpg.add_text("ManimTired")
            dpg.add_text(" - Explain complex concepts with AI-generated animations", color=[150, 150, 150])
        dpg.add_separator()
        # Prompt input and model selection
        dpg.add_text("Select LLM Model:")
        dpg.add_combo(["cerebras", "gpt4o", "o3"], default_value=args.llm, tag="llm_selector", width=150)
        dpg.add_text("Enter your animation prompt:")
        dpg.add_text("Request any educational concept to be explained with a Manim animation", color=[150, 150, 150])
        prompt_input = dpg.add_input_text(width=-1, height=100, multiline=True,
                                         hint="E.g., Explain quantum superposition; Show how binary search works; ...",
                                         tag="prompt_input")
        # Control buttons and status display
        with dpg.group(horizontal=True):
            dpg.add_button(label="Generate Animation", callback=on_generate, tag="generate_btn", width=200, height=30)
            def cancel_render():
                dpg.set_value("cancel_render", True)
                dpg.configure_item("status_text", default_value="Canceling render process...", color=[255, 165, 0])
            dpg.add_button(label="Cancel Render", callback=cancel_render, tag="cancel_btn", width=150, height=30, show=False)
            dpg.add_text("Status:", color=[150, 150, 150])
            dpg.add_text("Ready to generate", tag="status_text", color=[0, 255, 0])
        dpg.add_separator()
        # Loading/progress indicator
        with dpg.group(tag="loading_indicator", show=False):
            dpg.add_loading_indicator()
            dpg.add_text("Generating animation...", color=[255, 255, 0])
        # Warning message group (e.g., if using fallback)
        with dpg.group(tag="warning_group", show=False):
            dpg.add_text("", tag="warning_text", color=[255, 165, 0])
        # Error message group
        with dpg.group(tag="error_group", show=False):
            dpg.add_text("", tag="error_text", color=[255, 0, 0])
        # Result display group (video and controls)
        with dpg.group(tag="result_group", show=False):
            dpg.add_separator()
            dpg.add_text("Generated Animation:", color=[0, 255, 0])
            dpg.add_image(texture_tag="video_texture", tag="video_image", width=800, height=450)
            dpg.add_text("", tag="video_path_text")
            with dpg.group(horizontal=True):
                # Button to play within GUI
                def play_external():
                    try:
                        # Extract path from "Video path: X"
                        path_str = dpg.get_value("video_path_text").split(": ", 1)[1]
                        path = Path(path_str)
                        if path.exists():
                            os.startfile(str(path))
                        else:
                            dpg.configure_item("video_path_text", default_value="Error: Video file not found")
                    except Exception as e:
                        dpg.configure_item("video_path_text", default_value=f"Error opening video: {e}")
                def toggle_play_in_gui():
                    if dpg.get_value("is_playing"):
                        # Stop playback
                        dpg.set_value("is_playing", False)
                        dpg.configure_item("play_in_gui_btn", label="Play in GUI")
                    else:
                        try:
                            path_str = dpg.get_value("video_path_text").split(": ", 1)[1]
                            path = Path(path_str)
                            if path.exists():
                                play_video_in_gui(str(path))
                            else:
                                dpg.configure_item("video_path_text", default_value="Error: Video file not found")
                        except Exception as e:
                            dpg.configure_item("video_path_text", default_value=f"Error opening video: {e}")
                dpg.add_button(label="Play in GUI", callback=toggle_play_in_gui, width=150, tag="play_in_gui_btn")
                dpg.add_button(label="Play in External Player", callback=play_external, width=150)
                dpg.add_text(source="frame_counter")
                # Reset UI for new generation
                def reset_ui():
                    dpg.set_value("is_playing", False)
                    dpg.set_value("cancel_render", False)
                    dpg.configure_item("result_group", show=False)
                    dpg.configure_item("error_group", show=False)
                    dpg.configure_item("warning_group", show=False)
                    dpg.configure_item("cancel_btn", show=False)
                    dpg.configure_item("status_text", default_value="Ready to generate", color=[0, 255, 0])
                    dpg.configure_item("llm_selector", enabled=True)
                dpg.add_button(label="Generate Another Animation", callback=reset_ui, width=200)
    # Finalize and show GUI
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

# Callback for "Generate Animation" button
def on_generate():
    prompt = dpg.get_value("prompt_input")
    if not prompt.strip():
        dpg.configure_item("status_text", default_value="Please enter a prompt first.", color=[255, 0, 0])
        return
    # Reset any previous error/warning displays
    dpg.configure_item("error_group", show=False)
    dpg.configure_item("warning_group", show=False)
    # Prepare UI for generation process
    dpg.set_value("cancel_render", False)
    dpg.configure_item("cancel_btn", show=True)
    dpg.configure_item("prompt_input", enabled=False)
    dpg.configure_item("generate_btn", enabled=False)
    dpg.configure_item("llm_selector", enabled=False)
    dpg.configure_item("result_group", show=False)
    dpg.configure_item("loading_indicator", show=True)
    dpg.configure_item("status_text", default_value="Starting generation process...", color=[255, 255, 0])
    # Launch generation in a background thread
    def safe_generation():
        try:
            generation_thread(prompt)
        except Exception as e:
            err_msg = str(e).replace('\n', ' ').replace('"', "'")
            print(f"Unhandled error in generation thread: {err_msg}")
            dpg.configure_item("status_text", default_value=f"Fatal error: {err_msg}", color=[255, 0, 0])
            dpg.configure_item("video_path_text", default_value=f"Error: {err_msg}")
            dpg.configure_item("result_group", show=True)
            # Re-enable controls on fatal error
            dpg.configure_item("generate_btn", enabled=True)
            dpg.configure_item("prompt_input", enabled=True)
            dpg.configure_item("llm_selector", enabled=True)
            dpg.configure_item("loading_indicator", show=False)
            dpg.configure_item("cancel_btn", show=False)
    threading.Thread(target=safe_generation, daemon=True).start()

if __name__ == "__main__":
    main()
