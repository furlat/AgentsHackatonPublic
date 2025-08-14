# GigaReader

A humble attempt at extracting structured narrative actions from books using AI inference.

## Overview

This system processes narrative text from books to extract structured information about actions, interactions, and behaviors between entities. It's designed to identify and catalog the rich tapestry of actions that occur in stories - from physical movements to emotional changes.

# Data and Results:

the processed Gutenberg books can be found here `https://drive.google.com/drive/folders/1bUQqgw4jMzr1Sq3z9p39ari-5IpHAQcR?usp=drive_link` and they need to be copied in the `/data` folder.
while a subset of results can be found here `https://drive.google.com/drive/folders/1ycbZhwWTRlofnmJfoGkmWQecz3cUJBhP?usp=drive_link` and they need to be copied in the `/data/outs` folder for post-processing.

## System Architecture

### Core Components

**📖 Book Processor (`book_processor.py`)**
- Main processing engine that analyzes narrative text
- Extracts structured actions using AI inference
- Processes books in parallel for efficiency
- Handles batch processing with configurable limits

**🚀 Modal Server (`modal_servers/modal_qwen30b.py`)**
- Serverless inference endpoint using Modal
- Runs Qwen3-30B model via vLLM for fast inference
- OpenAI-compatible API for seamless integration
- Auto-scaling GPU infrastructure

### Processing Pipeline

1. **Text Ingestion**: Books are loaded and segmented into processable chunks
2. **AI Analysis**: Each segment is analyzed by the Qwen3-30B model to extract actions
3. **Structured Output**: Actions are formatted according to our schema
4. **Parallel Processing**: Multiple books processed concurrently for efficiency

## Data Schema

### NarrativeAnalysis Model

The core output structure for each analyzed text segment:

```python
class NarrativeAnalysis:
    text_id: str                    # Unique identifier for the text segment
    text_had_no_actions: bool       # Whether any actions were found (default: False)
    actions: List[Action]           # Ordered list of extracted actions
```

### Action Model

Each action captures a complete interaction between entities:

```python
class Action:
    # Source Entity
    source: str                     # Name of entity performing the action
    source_type: str               # Category: person, animal, object, location
    source_is_character: bool      # Whether source is a named character
    
    # Target Entity  
    target: str                    # Name of entity receiving the action
    target_type: str              # Category: person, animal, object, location
    target_is_character: bool     # Whether target is a named character
    
    # Action Details
    action: str                   # Verb describing the interaction
    consequence: str              # Immediate outcome or result
    
    # Text Evidence
    text_describing_the_action: str      # Original text fragment
    text_describing_the_consequence: str # Consequence description
    
    # Context
    location: str                 # Where the action occurs
    temporal_order_id: int        # Sequential ordering within the narrative
```

### Action Types Captured

The system identifies a broad range of actions:

- **Physical Interactions**: "Maya picked up the lantern"
- **Movements**: "The fox darted into the forest" 
- **Observable Behaviors**: "Professor Lin frowned"
- **Implied Actions**: "Sarah found herself tumbling" → "tumble"
- **Dialogue Actions**: "'I tossed it,' said Eliza" → "toss"
- **Mental Changes**: "She realized the truth" → "realize"
- **Emotional States**: "She felt sad" → "feel"

## Configuration

### Processing Parameters

- `book_start`: Starting book index for processing
- `num_books`: Number of books to process
- `max_calls`: Maximum API calls per book (optional)
- `max_batch_size`: Batch size for parallel processing
- `parallel_books`: Number of books to process concurrently

### Model Configuration

- **Model**: Qwen/Qwen3-30B-A3B-Instruct-2507
- **Endpoint**: Modal-hosted vLLM server with auto-scaling
- **Rate Limits**: 80 requests/minute, 200M tokens/minute
- **Engine**: vLLM V1 for improved performance

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup Instructions

1. **Clone the GigaReader repository**
   ```bash
   git clone https://github.com/furlat/GigaReader.git
   cd GigaReader
   ```

2. **Install MinInference framework dependencies**
   ```bash
   pip install -r MultiInference/requirements.txt
   ```

3. **Install MinInference in development mode**
   ```bash
   pip install -e MultiInference/
   ```

4. **Install additional dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory with your API keys if you want to use OpenAi but this demo is with Modal:
   ```bash
   # Add your API keys here
   OPENAI_API_KEY=your_openai_key_here
   # Add other provider keys as needed
   ```

### Verify Installation

You can verify the installation by running:
```python
from minference.threads.inference import InferenceOrchestrator
print("MinInference successfully installed!")
```

## Usage

### Basic Processing

```python
# Process a single book
await process_book(book_start=0, num_books=1)

# Process multiple books in parallel
await process_books_parallel(
    book_start=0, 
    num_books=5, 
    parallel_books=3,
    max_batch_size=32
)
```

### Server Deployment

The Modal server can be deployed with:

```bash
modal deploy modal_servers/modal_qwen30b.py
```

## Technical Details

### Inference Orchestration

- Uses `InferenceOrchestrator` for managing API calls and rate limits
- Structured output via Pydantic models ensures data consistency
- Error handling and validation for robust processing

### Performance Optimizations

- **Parallel Processing**: Multiple books processed simultaneously
- **Batch Processing**: Efficient batching of text segments
- **GPU Acceleration**: vLLM with CUDA optimization
- **Caching**: Model weights and compilation artifacts cached

## Dependencies

### Core Framework
- **`minference`**: Custom inference orchestration framework (included in `MultiInference/` directory)
  - Handles parallel API calls and rate limiting
  - Provides structured output via Pydantic models
  - Manages LLM client configurations and threading

### External Libraries
- `polars`: Fast data processing for handling book data
- `modal`: Serverless GPU infrastructure for model hosting
- `vllm`: High-performance LLM serving with CUDA optimization
- `pydantic`: Data validation and serialization for structured outputs
- `asyncio`: Asynchronous processing for parallel book analysis
- `dotenv`: Environment variable management

## Humble Notes

This system represents our best effort to systematically extract the rich narrative structure from books. While we've tried to be comprehensive in capturing actions, we recognize that narrative analysis is inherently complex and subjective. The system errs on the side of inclusion rather than exclusion when identifying actions, believing that capturing more context is generally preferable to missing important interactions.

We hope this tool proves useful for researchers, writers, and anyone interested in understanding the structural elements of storytelling.
