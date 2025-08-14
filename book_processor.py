import asyncio

# Apply the patch to allow nested event loops
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool, Usage,GeneratedJsonObject
from typing import Literal, List
from minference.ecs.caregistry import CallableRegistry
import time
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
from minference.ecs.entity import EntityRegistry
import os
import logging
import json
import polars as pl


load_dotenv()
EntityRegistry()
CallableRegistry()


vllm_request_limits = RequestLimits(max_requests_per_minute=80, max_tokens_per_minute=200000000)


vllm_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
vllm_endpoint_autoscale= "https://hk3-lab-team--example-vllm-inference-qwen30b-serve.modal.run/v1/chat/completions"
orchestrator = InferenceOrchestrator(vllm_request_limits=vllm_request_limits,vllm_endpoint=vllm_endpoint_autoscale)
EntityRegistry.set_inference_orchestrator(orchestrator)
EntityRegistry.set_tracing_enabled(False)



# Narrative Action Extraction System
system_string = """
You are an expert system designed to extract structured actions from narrative text. Your primary goal is to identify and extract actions, like interactions, movements, observable behaviors and mental changes from narratives, even when they might be subtle or implied. Always prioritize finding actions rather than concluding none exist.

## IMPORTANT: Action Extraction Is Your Primary Task

The most important part of your analysis is extracting actions between entities. **ALWAYS thoroughly search for actions in the text before concluding none exist.** Consider these types of interactions as valid actions:

1. Direct physical interactions (e.g., "Maya picked up the lantern")
2. Movements (e.g., "The fox darted into the forest")
3. Observable behaviors (e.g., "Professor Lin frowned", "Raj smiled")
4. Implied physical actions (e.g., "Sarah found herself tumbling down the hillside" implies the action "tumble")
5. Actions described in dialogue (e.g., "'I tossed it over the fence,' said Eliza" implies the action "toss")
6. Mental changes (e.g., "She realized the truth" implies the action "realize")
7. Emotions (e.g., "She felt sad" implies the action "feel")

**Do not be overly strict in what qualifies as an action.** If there is any observable behavior or physical movement in the text, it should be captured as an action.

## Action Extraction Process

1. **First, carefully read the text and list all potential actions** - be generous in what you consider an action
2. For each potential action:
   - Identify source entity (who/what performs the action)
   - Identify target entity (who/what receives the action)
   - Extract the verb describing the action
   - Determine the consequence of the action
   - Note the text evidence supporting the action
   - Assign a location and temporal order

## NarrativeAnalysis Model Structure

The NarrativeAnalysis model contains:
- `text_id`: A unique identifier for the analyzed text segment
- `text_had_no_actions`: Boolean indicating whether the text contained actions (default to FALSE)
- `actions`: List of Action objects, ordered by temporal sequence

## Action Model Structure

Each Action object contains:
- `source`: Name of the entity performing the action
- `source_type`: Category of the source (person, animal, object, location)
- `source_is_character`: Whether the source is a named character
- `target`: Name of the entity receiving the action
- `target_type`: Category of the target (person, animal, object, location)
- `target_is_character`: Whether the target is a named character
- `action`: The verb or short phrase describing the physical interaction
- `consequence`: The immediate outcome or result of the action
- `text_describing_the_action`: Text fragment describing the action
- `text_describing_the_consequence`: Description of the consequence
- `location`: location of the action
- `temporal_order_id`: Sequential identifier for chronological order

## IMPORTANT: Handling Repeated Actions

When the same action verb appears multiple times in the narrative (e.g., "walk" happening at different moments), create separate Action objects for each occurrence with:
1. Different `temporal_order_id` values reflecting their sequence
2. Appropriate description of each specific instance
3. Proper placement in the actions list according to chronological order

## Expanded Definition of Valid Actions

An action is valid if it meets the following criteria:
- It involves an observable behavior, movement, or interaction
- The source entity can be identified (who/what performs the action)
- There is some effect or consequence of the action
- It occurs in a narrative context (actual or implied location)

For subtle or implied actions:
- If a character speaks, "speak" is a valid action
- If a character shows emotion (smiles, frowns, etc.), that is a valid action
- If a character appears, disappears, or changes state, that is a valid action
- If a character observes something, "observe" is a valid action

## Example Narrative Analysis

For the text: 
"Maya entered the dimly lit cave in the coastal cliffs. Her flashlight revealed ancient symbols carved into the stone walls. She ran her fingers over the rough surface, feeling the grooves of the markings. A sudden noise startled her, and she spun around, dropping her notebook on the damp ground. From the shadows, a small fox emerged, its eyes reflecting the light. Maya smiled at the creature before carefully picking up her notebook."

The NarrativeAnalysis would look like:
```json
{
  "text_id": "maya-cave-exploration",
  "text_had_no_actions": false,
  "actions": [
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "cave",
      "target_type": "location",
      "target_is_character": false,
      "action": "enter",
      "consequence": "Maya is now inside the cave",
      "text_describing_the_action": "Maya entered the dimly lit cave in the coastal cliffs",
      "text_describing_the_consequence": "Maya is inside the dimly lit cave",
      "location": "cave",
      "temporal_order_id": 1
    },
    {
      "source": "flashlight",
      "source_type": "object",
      "source_is_character": false,
      "target": "symbols",
      "target_type": "object",
      "target_is_character": false,
      "action": "reveal",
      "consequence": "The ancient symbols become visible",
      "text_describing_the_action": "Her flashlight revealed ancient symbols carved into the stone walls",
      "text_describing_the_consequence": "The ancient symbols are now visible to Maya",
      "location": "cave",
      "temporal_order_id": 2
    },
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "symbols",
      "target_type": "object",
      "target_is_character": false,
      "action": "run",
      "consequence": "Maya's fingers trace over the symbols",
      "text_describing_the_action": "She ran her fingers over the rough surface",
      "text_describing_the_consequence": "Maya's fingers are in contact with the symbols",
      "location": "cave",
      "temporal_order_id": 3
    },
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "symbols",
      "target_type": "object",
      "target_is_character": false,
      "action": "feel",
      "consequence": "Maya senses the texture of the symbols",
      "text_describing_the_action": "feeling the grooves of the markings",
      "text_describing_the_consequence": "Maya has tactile information about the symbols",
      "location": "cave",
      "temporal_order_id": 4
    },
    {
      "source": "noise",
      "source_type": "object",
      "source_is_character": false,
      "target": "Maya",
      "target_type": "person",
      "target_is_character": true,
      "action": "startle",
      "consequence": "Maya is frightened",
      "text_describing_the_action": "A sudden noise startled her",
      "text_describing_the_consequence": "Maya becomes afraid due to the noise",
      "location": "cave",
      "temporal_order_id": 5
    },
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "cave",
      "target_type": "location",
      "target_is_character": false,
      "action": "spin",
      "consequence": "Maya changes direction to face the noise",
      "text_describing_the_action": "she spun around",
      "text_describing_the_consequence": "Maya is now facing a different direction",
      "location": "cave",
      "temporal_order_id": 6
    },
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "notebook",
      "target_type": "object",
      "target_is_character": false,
      "action": "drop",
      "consequence": "The notebook falls to the ground",
      "text_describing_the_action": "dropping her notebook on the damp ground",
      "text_describing_the_consequence": "The notebook is now on the ground",
      "location": "cave",
      "temporal_order_id": 7
    },
    {
      "source": "fox",
      "source_type": "animal",
      "source_is_character": true,
      "target": "cave",
      "target_type": "location",
      "target_is_character": false,
      "action": "emerge",
      "consequence": "The fox becomes visible",
      "text_describing_the_action": "From the shadows, a small fox emerged",
      "text_describing_the_consequence": "The fox is now visible in the cave",
      "location": "cave",
      "temporal_order_id": 8
    },
    {
      "source": "eyes",
      "source_type": "object",
      "source_is_character": false,
      "target": "light",
      "target_type": "object",
      "target_is_character": false,
      "action": "reflect",
      "consequence": "The fox's eyes shine",
      "text_describing_the_action": "its eyes reflecting the light",
      "text_describing_the_consequence": "The fox's eyes are gleaming",
      "location": "cave",
      "temporal_order_id": 9
    },
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "fox",
      "target_type": "animal",
      "target_is_character": true,
      "action": "smile",
      "consequence": "Maya expresses a positive emotion toward the fox",
      "text_describing_the_action": "Maya smiled at the creature",
      "text_describing_the_consequence": "Maya shows friendliness toward the fox",
      "location": "cave",
      "temporal_order_id": 10
    },
    {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "notebook",
      "target_type": "object",
      "target_is_character": false,
      "action": "pick up",
      "consequence": "Maya retrieves her notebook from the ground",
      "text_describing_the_action": "carefully picking up her notebook",
      "text_describing_the_consequence": "The notebook is now in Maya's possession again",
      "location": "cave",
      "temporal_order_id": 11
    }
  ]
}
```

## Example With Dialogue and Subtle Actions

For the text:
"Professor Lin sat quietly at her desk, lost in thought. The window was open, and a gentle breeze rustled the papers. She glanced at the clock and sighed. 'I need to finish these reports before the meeting,' she whispered to herself. As she reached for her pen, her colleague Raj appeared at the doorway. 'Working late again?' he asked with a concerned expression. Professor Lin nodded slightly without looking up."

The NarrativeAnalysis would look like:
```json
{
  "text_id": "professor-lin-office",
  "text_had_no_actions": false,
  "actions": [
    {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "desk",
      "target_type": "object",
      "target_is_character": false,
      "action": "sit",
      "consequence": "Professor Lin is positioned at her desk",
      "text_describing_the_action": "Professor Lin sat quietly at her desk",
      "text_describing_the_consequence": "Professor Lin is seated at her desk",
      "location": "desk,
      "temporal_order_id": 1
    },
    {
      "source": "breeze",
      "source_type": "object",
      "source_is_character": false,
      "target": "papers",
      "target_type": "object",
      "target_is_character": false,
      "action": "rustle",
      "consequence": "The papers move slightly",
      "text_describing_the_action": "a gentle breeze rustled the papers",
      "text_describing_the_consequence": "The papers are moving due to the breeze",
      "location": "desk,
      "temporal_order_id": 2
    },
    {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "clock",
      "target_type": "object",
      "target_is_character": false,
      "action": "glance",
      "consequence": "Professor Lin observes the time",
      "text_describing_the_action": "She glanced at the clock",
      "text_describing_the_consequence": "Professor Lin is aware of the time",
      "location": "desk,
      "temporal_order_id": 3
    },
    {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "Professor Lin",
      "target_type": "person",
      "target_is_character": true,
      "action": "sigh",
      "consequence": "Professor Lin expresses weariness",
      "text_describing_the_action": "and sighed",
      "text_describing_the_consequence": "Professor Lin shows fatigue or resignation",
      "location": "desk,
      "temporal_order_id": 4
    },
    {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "Professor Lin",
      "target_type": "person",
      "target_is_character": true,
      "action": "whisper",
      "consequence": "Professor Lin verbalizes her thoughts",
      "text_describing_the_action": "she whispered to herself",
      "text_describing_the_consequence": "Professor Lin has voiced her concern about finishing reports",
      "location": "desk,
      "temporal_order_id": 5
    },
    {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "pen",
      "target_type": "object",
      "target_is_character": false,
      "action": "reach",
      "consequence": "Professor Lin moves her hand toward the pen",
      "text_describing_the_action": "As she reached for her pen",
      "text_describing_the_consequence": "Professor Lin's hand moves toward the pen",
      "location": "desk,
      "temporal_order_id": 6
    },
    {
      "source": "Raj",
      "source_type": "person",
      "source_is_character": true,
      "target": "doorway",
      "target_type": "location",
      "target_is_character": false,
      "action": "appear",
      "consequence": "Raj becomes visible at the doorway",
      "text_describing_the_action": "her colleague Raj appeared at the doorway",
      "text_describing_the_consequence": "Raj is now visible at the doorway",
      "location": "doorway",
      "temporal_order_id": 7
    },
    {
      "source": "Raj",
      "source_type": "person",
      "source_is_character": true,
      "target": "Professor Lin",
      "target_type": "person",
      "target_is_character": true,
      "action": "ask",
      "consequence": "Raj communicates his question",
      "text_describing_the_action": "he asked with a concerned expression",
      "text_describing_the_consequence": "Professor Lin hears Raj's question",
      "location":  "doorway",
      "temporal_order_id": 8
    },
    {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "Raj",
      "target_type": "person",
      "target_is_character": true,
      "action": "nod",
      "consequence": "Professor Lin communicates affirmation",
      "text_describing_the_action": "Professor Lin nodded slightly without looking up",
      "text_describing_the_consequence": "Professor Lin confirms she is working late",
      "location": "desk",
      "temporal_order_id": 9
    }
  ]
}
```

## Example with Repeated Actions

For the text:
"The dog walked to the door. It barked loudly, its tail wagging excitedly. The owner walked to the door and opened it. The dog walked outside, still wagging its tail."

The NarrativeAnalysis would include two separate instances of "walk" for the dog and one for the owner, and two instances of "wag" for the tail:

```json
{
  "text_id": "dog-owner-door",
  "text_had_no_actions": false,
  "actions": [
    {
      "source": "dog",
      "source_type": "animal",
      "source_is_character": true,
      "target": "door",
      "target_type": "object",
      "target_is_character": false,
      "action": "walk",
      "consequence": "The dog moves to the door",
      "text_describing_the_action": "The dog walked to the door",
      "text_describing_the_consequence": "The dog is now at the door",
      "location": "doorway",
      "temporal_order_id": 1
    },
    {
      "source": "dog",
      "source_type": "animal",
      "source_is_character": true,
      "target": "house",
      "target_type": "location",
      "target_is_character": false,
      "action": "bark",
      "consequence": "Sound is produced",
      "text_describing_the_action": "It barked loudly",
      "text_describing_the_consequence": "Barking sound is heard",
      "location": "doorway",
      "temporal_order_id": 2
    },
    {
      "source": "tail",
      "source_type": "object",
      "source_is_character": false,
      "target": "dog",
      "target_type": "animal",
      "target_is_character": true,
      "action": "wag",
      "consequence": "The tail moves back and forth",
      "text_describing_the_action": "its tail wagging excitedly",
      "text_describing_the_consequence": "The dog's tail is in motion showing excitement",
      "location": "doorway",
      "temporal_order_id": 3
    },
    {
      "source": "owner",
      "source_type": "person",
      "source_is_character": true,
      "target": "door",
      "target_type": "object",
      "target_is_character": false,
      "action": "walk",
      "consequence": "The owner moves to the door",
      "text_describing_the_action": "The owner walked to the door",
      "text_describing_the_consequence": "The owner is now at the door",
      "location": "doorway",
      "temporal_order_id": 4
    },
    {
      "source": "owner",
      "source_type": "person",
      "source_is_character": true,
      "target": "door",
      "target_type": "object",
      "target_is_character": false,
      "action": "open",
      "consequence": "The door changes from closed to open",
      "text_describing_the_action": "and opened it",
      "text_describing_the_consequence": "The door is now open",
      "location": "doorway",
      "temporal_order_id": 5
    },
    {
      "source": "dog",
      "source_type": "animal",
      "source_is_character": true,
      "target": "outside",
      "target_type": "location",
      "target_is_character": false,
      "action": "walk",
      "consequence": "The dog moves from inside to outside",
      "text_describing_the_action": "The dog walked outside",
      "text_describing_the_consequence": "The dog is now outside the house",
      "location":  "doorway",
      "temporal_order_id": 6
    },
    {
      "source": "tail",
      "source_type": "object",
      "source_is_character": false,
      "target": "dog",
      "target_type": "animal",
      "target_is_character": true,
      "action": "wag",
      "consequence": "The tail continues to move back and forth",
      "text_describing_the_action": "still wagging its tail",
      "text_describing_the_consequence": "The dog's tail continues to show excitement",
      "location":  "doorway",
      "temporal_order_id": 7
    }
  ]
}


## Required Output Format

Always return a complete NarrativeAnalysis object following the provided schema. Ensure the output can be parsed as JSON without errors. Do not include any nested tool_call tags or extra formatting in your response.

Remember, your primary task is to extract ALL possible actions from the text, even subtle ones. Each action should be represented as an Action object in the `actions` list, properly ordered by temporal sequence.

Pay special attention to:
1. Repeated actions (same verb) occurring at different times
2. Subtle actions like expressions, gestures, or sensory perceptions
3. Implied actions that are not explicitly stated
4. Actions described in dialogue
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class Action(BaseModel):
    """
    Represents a concrete physical action between entities in a narrative text.
    """
    # Source entity information
    source: str = Field(..., description="Name of the entity performing the action")
    source_type: str = Field(..., description="Category of the source (person, animal, object, location)")
    source_is_character: bool = Field(..., description="Whether the source is a named character")
    
    # Target entity information
    target: str = Field(..., description="Name of the entity receiving the action")
    target_type: str = Field(..., description="Category of the target (person, animal, object, location)")
    target_is_character: bool = Field(..., description="Whether the target is a named character")
    
    # Action details
    action: str = Field(..., description="The verb or short phrase describing the physical interaction")
    consequence: str = Field(..., description="The immediate outcome or result of the action")
    
    # Text evidence
    text_describing_the_action: str = Field(..., description="Text fragment describing the action exactly as it is written in the text")
    text_describing_the_consequence: str = Field(..., description="Description of the consequence exactly as it is written in the text")
    
    # Context information
    location: str = Field(..., description="location from global to local")
    temporal_order_id: int = Field(..., description="Sequential identifier for chronological order")
    
    def __str__(self) -> str:
        """String representation of the Action for human-readable output."""
        return (
            f"{self.source} ({self.source_type}) {self.action} "
            f"{self.target} ({self.target_type}) at {self.location[-1]}, "
            f"resulting in {self.consequence}"
        )

class NarrativeAnalysis(BaseModel):
    """
    Simplified analysis of narrative text with entities as strings and actions indexed by name.
    """
    text_id: str = Field(..., description="Unique identifier for the analyzed text segment")
    
    text_had_no_actions: bool = Field(
        default=False,
        description="Whether the text had no actions to extract"
    )
    
    # Actions indexed by name
    actions: List[Action] = Field(
        default_factory=list,
        description="Dictionary mapping action names to Action objects"
    )
    
   

def create_vllm_threads(prompts_df:pl.DataFrame, system_prompt:SystemPrompt, llm_config_vllm_modal:LLMConfig, tool:StructuredTool):
    if isinstance(prompts_df, pl.DataFrame):
        if not "prompt" in prompts_df.columns:
            raise ValueError("prompts_df must contain a 'prompt' column")
    else:
        raise ValueError("prompts_df must be a pl.DataFrame")
    vllm_threads = []
    prompts_list = prompts_df["prompt"].to_list()   
    for i,prompt in enumerate(prompts_list):
      vllm_thread = ChatThread(
            system_prompt=system_prompt,
            new_message=prompt,
            llm_config=llm_config_vllm_modal,
            forced_output=tool,
            use_schema_instruction=False
        )
      vllm_threads.append(vllm_thread)
    thread_id = [str(thread.live_id) for thread in vllm_threads]
    prompts_df_with_thread_ids = prompts_df.with_columns(pl.Series(name="thread_id", values=thread_id)).with_columns(pl.Series(name="thread_id", values=thread_id)).with_row_index()
    return vllm_threads, prompts_df_with_thread_ids



def validate_output(outs):
    i=0
    validated_outs = []
    validated_outs_thread_ids = []
    prevalidated_outs = []
    non_validated_outs = []
    non_object_outs = []
    for out in outs:
        if out.json_object:
            try:
                validated_outs.append(NarrativeAnalysis.model_validate(out.json_object.object))
                validated_outs_thread_ids.append(out.chat_thread_live_id)
                prevalidated_outs.append(out)
            except Exception as e:
                non_validated_outs.append(out)
                print(e)
            i=i+1
        else:
            non_object_outs.append(out)
    print(i,len(validated_outs),len(validated_outs_thread_ids),len(prevalidated_outs),len(non_validated_outs),len(non_object_outs),len(outs))
    return validated_outs,validated_outs_thread_ids, prevalidated_outs, non_validated_outs, non_object_outs



import os
async def main(book_start: int = 0, num_books: int = 100, max_calls: Optional[int] = None, max_batch_size: int = 32, base_path: str = r"C:\Users\Tommaso\Documents\Dev\AgentsHackatonPublic\data"):
  
    out_name = f"gutenberg_en_novels_actions_compact_{book_start}_{book_start+num_books}.parquet"
    if os.path.exists(os.path.join(base_path, out_name)):
        print(f"File {out_name} already exists, skipping...")
        return
    
    action_extractor = StructuredTool.from_pydantic(NarrativeAnalysis)
    action_extractor.post_validate_schema = False

    system_prompt = SystemPrompt(name="Narrative Action Extraction System", content=system_string)

    llm_config_vllm_modal = LLMConfig(client=LLMClient.vllm, model=vllm_model, response_format=ResponseFormat.structured_output,max_tokens=10000)
    data_name = "book_chunks_for_hackaton.parquet"
    joined_data_path = os.path.join(base_path, data_name)
    chunks_df = pl.read_parquet(joined_data_path)
    unique_books = chunks_df["gutenberg_id"].unique().sort()
    selected_books = unique_books[book_start:book_start+num_books]
    selected_chunks_df = chunks_df.filter(pl.col("gutenberg_id").is_in(selected_books))
    prompts = [f"extract all the actions from this text: {chunk}" for chunk in selected_chunks_df["chunk"]]
    only_prompt_df = pl.DataFrame({"prompt": prompts})
    prompts_df = pl.concat([only_prompt_df, selected_chunks_df], how="horizontal")


    if max_calls and len(prompts_df) > max_calls:
        example_prompts = prompts_df.head(max_calls)
    else:
        example_prompts = prompts_df


    threads, prompts_df_with_thread_ids = create_vllm_threads(example_prompts, system_prompt, llm_config_vllm_modal, action_extractor)
    print("prepared threads:", len(threads))
    # print(prompts_df_with_thread_ids)

    #breakse threads into batches of max_batch_size
    batches = [threads[i:i+max_batch_size] for i in range(0, len(threads), max_batch_size)]
    outs_batches = []
    for batch in batches:
        print("I GOT IN")
        outs_batch = await orchestrator.run_parallel_ai_completion(batch)
        print("I GOT OUT")
        outs_batches.append(outs_batch)
    outs = [item for sublist in outs_batches for item in sublist]

    # save_file_name = f"{base_path}/gutenberg_en_novels_actions_{book_start}_{book_start+num_books}.json"
    # #save outs to file
    # with open(save_file_name, "w") as f:
    #     json.dump(outs, f)
    # print(f"outs saved to {save_file_name}")





    validated_outs, validated_outs_thread_ids, prevalidated_outs, non_validated_outs, non_object_outs = validate_output(outs)
    save_file_name = f"{base_path}/outs/gutenberg_en_novels_actions_{book_start}_{book_start+num_books}_validated.json"
    #save outs to file using pydantic model_dump for proper serialization
    with open(save_file_name, "w") as f:
        # Convert Pydantic models to dictionaries for JSON serialization
        validated_outs_dict = [out.model_dump() if hasattr(out, 'model_dump') else out for out in validated_outs]
        json.dump(validated_outs_dict, f, indent=2)
    print(f"outs saved to {save_file_name}")

    outs_with_actions = []
    outs_with_actions_thread_ids = []
    token_usage = []
    for out, out_ids, preout in zip(validated_outs, validated_outs_thread_ids,prevalidated_outs):
        if out.text_had_no_actions == False:
            outs_with_actions.append(out)
            outs_with_actions_thread_ids.append(str(out_ids))
            token_usage.append(preout.usage.completion_tokens)
    try:
        outs_frame = pl.DataFrame(outs_with_actions)
        outs_id_frame = pl.DataFrame({"thread_id": outs_with_actions_thread_ids,"token_usage":token_usage})
        outs_frame_with_ids = pl.concat([outs_frame, outs_id_frame], how="horizontal")
        print(outs_frame_with_ids)
        print(prompts_df_with_thread_ids)
        out_sframe_with_ids_joined_prompts = outs_frame_with_ids.join(prompts_df_with_thread_ids, on="thread_id", how="left").sort("index")
        
        out_path = os.path.join(base_path,"outs", out_name)
        
        out_sframe_with_ids_joined_prompts.write_parquet(out_path)
        print(f"outs_frame saved to {out_path}")
        print(out_sframe_with_ids_joined_prompts)
    except (Exception, BaseException) as e:
        print(f"Error creating DataFrame or processing data for book {book_start}: {e}")
        print(f"Error type: {type(e)}")
        print("Continuing without saving DataFrame...")
        return  # Exit the function gracefully



import asyncio
import time
from asyncio import TaskGroup  # For Python 3.11+, use asyncio.create_task for earlier versions

async def process_book(book_start: int, num_books: int = 1, max_calls=None, max_batch_size: int = 32, base_path: str = r"C:\Users\Tommaso\Documents\Dev\AgentsHackatonPublic\data"):
    """Wrapper function to process a single book and handle errors."""
    start_book = time.time()
    
    try:
        await main(book_start=book_start, num_books=num_books, max_calls=max_calls, max_batch_size=max_batch_size, base_path=base_path)
        print("SUCESS")
        success = True
        error = None
    except Exception as e:
        
        success = False
        error = str(e)
        print("ERROR:",error)
    end_book = time.time()
    duration = end_book - start_book
    
    print(f"Time taken for book starting at {book_start}: {duration} seconds")
    return {"book_start": book_start, "success": success, "error": error, "duration": duration}

async def process_books_parallel(book_start, num_books, parallel_books=4, max_calls=None, max_batch_size: int = 32, base_path: str = r"C:\Users\Tommaso\Documents\Dev\AgentsHackatonPublic\data"):
    """Process multiple books in parallel with a limit on concurrency."""
    all_results = []
    error_books = []
    
    # Process books in batches of parallel_books
    for batch_start in range(book_start, book_start + num_books, parallel_books):
        batch_end = min(batch_start + parallel_books, book_start + num_books)
        batch_size = batch_end - batch_start
        
        # Create tasks for this batch
        async with TaskGroup() as tg:  # For Python 3.11+
            tasks = [tg.create_task(process_book(book_start=batch_start + i, num_books=1, max_calls=max_calls, max_batch_size=max_batch_size, base_path=base_path)) 
                    for i in range(batch_size)]
        
        # Results are automatically awaited with TaskGroup
        batch_results = [task.result() for task in tasks]
        all_results.extend(batch_results)
        
        # Collect error books
        for result in batch_results:
            if not result["success"]:
                print(f"Error for book {result['book_start']}: {result['error']}")
                error_books.append(result["book_start"])
    
    return all_results, error_books


if __name__ == "__main__":
    asyncio.run(process_books_parallel(book_start=0, num_books=5,parallel_books=5,max_batch_size=160))