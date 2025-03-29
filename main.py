import os
import threading
import time
from pathlib import Path
import dearpygui.dearpygui as dpg
from dotenv import load_dotenv
import argparse
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import subprocess
import re
from typing import Optional, Tuple

# Parse CLI args
parser = argparse.ArgumentParser(description="ManimTired - AI-powered Manim animations")
parser.add_argument("--llm", choices=["cerebras", "gpt4o"], default="cerebras", help="LLM to use for code generation")
args = parser.parse_args()

# Load environment variables
load_dotenv()
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not CEREBRAS_BASE_URL and args.llm == "cerebras":
    raise ValueError("CEREBRAS_BASE_URL not found in .env")
if not OPENAI_BASE_URL and args.llm == "gpt4o":
    raise ValueError("OPENAI_BASE_URL not found in .env")

# Initialize paths
TEMP_SCRIPTS_DIR = Path("temp_scripts")
GENERATED_VIDEOS_DIR = Path("generated_videos")
TEMP_SCRIPTS_DIR.mkdir(exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(exist_ok=True)

# Initialize LLM, parser, and prompt
class ManimCode(BaseModel):
    """Schema for Manim code generation output."""
    code: str = Field(description="The complete Python code for the Manim scene")

parser = JsonOutputParser(pydantic_object=ManimCode)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Manim code generator. You MUST output ONLY valid JSON matching this schema:
{format_instructions}

Your code MUST:
1. Define a single Manim scene class named `GeneratedScene`.
2. Place all titles, explanatory text, and labels **on the left side of the screen** using `.to_edge(LEFT)` or `.shift(LEFT * X)`.
3. Place all diagrams, shapes, graphs, and animations **on the right side of the screen** using `.to_edge(RIGHT)` or `.shift(RIGHT * X)`.
4. Ensure text and diagrams never overlap. Use `.shift` if necessary to adjust spacing.
5. Maintain logical alignment between text and diagrams where relevant (e.g., a label should align with its corresponding graph).
6. Keep the scene simple and clean. Do not overcrowd.
7. Only use valid Manim classes and functions. Use standard LaTeX notation in MathTex (e.g., MathTex(r"\\frac{{1}}{{2}}")).
8. Do not create custom classes, functions, or imports. Stick to official Manim functions.
9. For degree symbols, use LaTeX notation (\\degree) in MathTex or the Unicode degree symbol (°) in Text with proper encoding.
    """),
    ("human", "{prompt}")
])

# Initialize LLM based on CLI arg
if args.llm == "cerebras":
    llm = ChatCerebras(
        model_name="llama-3.3-70b",
        base_url=CEREBRAS_BASE_URL,
        temperature=0.7,
        max_tokens=2000,
    )
else:  # gpt4o
    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=OPENAI_BASE_URL,
        temperature=0.7,
        max_tokens=2000,
    )

# Create the chain
chain = prompt | llm | parser

def generate_manim_code(user_prompt: str) -> str:
    """Generate Manim code from a prompt using Cerebras LLM."""
    try:
        response = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "prompt": user_prompt
        })
        
        # Response is a dict since we're using JsonOutputParser
        code = response["code"]
        
        # Basic validation
        if not "class GeneratedScene" in code or not "Scene" in code:
            raise ValueError("Generated code doesn't look like a valid Manim scene")
        
        return code
    except Exception as e:
        if "Invalid json output" in str(e):
            raise ValueError("LLM failed to generate valid code. Try rephrasing your prompt to be more specific about the animation you want.")
        raise

def run_manim_render(script_path: str, timestamp: str) -> Tuple[bool, Optional[str], str, str]:
    """Run Manim to generate video and thumbnail. Returns (success, error_msg, video_path, thumb_path)."""
    try:
        # Render video
        subprocess.run([
            "manim", "-ql", script_path, "GeneratedScene",
            "--media_dir", "generated_videos",
            "-o", f"video_{timestamp}.mp4"
        ], check=True)
        
        # Render thumbnail (just the last frame)
        subprocess.run([
            "manim", "-ql", "--format", "png", script_path, "GeneratedScene",
            "--media_dir", "generated_videos",
            "-o", f"thumb_{timestamp}.png"
        ], check=True)
        
        # Find the actual video file (Manim creates nested directories)
        video_dir = GENERATED_VIDEOS_DIR / "videos" / f"script_{timestamp}" / "480p15"
        video_path = video_dir / f"video_{timestamp}.mp4"
        
        # Find the actual thumbnail (Manim creates a directory of frames)
        thumb_dir = GENERATED_VIDEOS_DIR / "images" / f"script_{timestamp}"
        thumb_files = list(thumb_dir.glob("*.png"))
        if not thumb_files:
            raise ValueError("No thumbnail generated")
        thumb_path = thumb_files[0]  # Use first frame as thumbnail
        
        if not video_path.exists():
            raise ValueError(f"Video not found at {video_path}")
        
        return True, None, str(video_path), str(thumb_path)
    except Exception as e:
        return False, f"Manim render failed: {str(e)}", None, None

def generation_thread(prompt: str):
    """Background thread for handling generation without blocking UI."""
    try:
        timestamp = str(int(time.time()))
        script_path = TEMP_SCRIPTS_DIR / f"script_{timestamp}.py"
        
        # Generate code
        code = generate_manim_code(prompt)
        
        # Save to temp file
        script_path.write_text(code)
        
        # Render
        success, error, video_path, thumb_path = run_manim_render(str(script_path), timestamp)
        
        # Cleanup temp script
        script_path.unlink()
        
        if success and video_path and thumb_path:
            # Update UI from thread
            dpg.configure_item("generate_btn", enabled=True)
            dpg.configure_item("prompt_input", enabled=True)
            dpg.configure_item("loading_indicator", show=False)
            
            try:
                # Load and show thumbnail
                width, height, channels, data = dpg.load_image(thumb_path)
                texture_id = f"thumb_{timestamp}"
                dpg.add_static_texture(
                    width, height, data,
                    parent="texture_registry",
                    tag=texture_id
                )
                dpg.configure_item("result_group", show=True)
                dpg.configure_item("result_image", texture_tag=texture_id)
                dpg.configure_item("video_path_text", default_value=f"Video saved to: {video_path}")
                
                # Try to open the video
                try:
                    os.startfile(video_path)
                except:
                    pass  # Fallback to manual open button
            except Exception as e:
                dpg.configure_item("result_group", show=True)
                dpg.configure_item("video_path_text", default_value=f"Video generated but failed to load preview: {str(e)}")
        else:
            # Show error
            dpg.configure_item("generate_btn", enabled=True)
            dpg.configure_item("prompt_input", enabled=True)
            dpg.configure_item("loading_indicator", show=False)
            dpg.configure_item("result_group", show=True)
            dpg.configure_item("video_path_text", default_value=f"Error: {error}")
            
    except Exception as e:
        # Show error
        dpg.configure_item("generate_btn", enabled=True)
        dpg.configure_item("prompt_input", enabled=True)
        dpg.configure_item("loading_indicator", show=False)
        dpg.configure_item("result_group", show=True)
        dpg.configure_item("video_path_text", default_value=f"Error: {str(e)}")

def main():
    dpg.create_context()
    dpg.create_viewport(title="ManimTired", width=800, height=600)
    
    with dpg.texture_registry(tag="texture_registry"):
        # Create a blank 1x1 black texture as placeholder
        dpg.add_static_texture(
            1, 1, [0, 0, 0, 255],
            parent="texture_registry",
            tag="blank_texture"
        )
    
    with dpg.window(label="ManimTired", tag="primary"):
        dpg.add_text("Enter your animation prompt:")
        prompt_input = dpg.add_input_text(
            width=-1,
            height=100,
            multiline=True,
            hint="Describe the animation you want to create...",
            tag="prompt_input"
        )
        
        def on_generate():
            prompt = dpg.get_value(prompt_input)
            if not prompt.strip():
                return
            
            # Disable input while generating
            dpg.configure_item("prompt_input", enabled=False)
            dpg.configure_item("generate_btn", enabled=False)
            dpg.configure_item("result_group", show=False)
            dpg.configure_item("loading_indicator", show=True)
            
            # Start generation in background
            thread = threading.Thread(
                target=generation_thread,
                args=(prompt,)
            )
            thread.start()
        
        dpg.add_button(
            label="Generate Animation",
            callback=on_generate,
            tag="generate_btn"
        )
        
        with dpg.group(tag="loading_indicator", show=False):
            dpg.add_loading_indicator()
            dpg.add_text("Generating animation...")
        
        with dpg.group(tag="result_group", show=False):
            dpg.add_image(
                texture_tag="blank_texture",
                tag="result_image"
            )
            dpg.add_text("", tag="video_path_text")
            
            def open_video():
                try:
                    path = dpg.get_value("video_path_text").split(": ", 1)[1]
                    if os.path.exists(path):
                        os.startfile(path)
                    else:
                        dpg.configure_item("video_path_text", default_value="Error: Video file not found")
                except Exception as e:
                    dpg.configure_item("video_path_text", default_value=f"Error opening video: {str(e)}")
            
            dpg.add_button(
                label="Open Video",
                callback=open_video
            )
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()