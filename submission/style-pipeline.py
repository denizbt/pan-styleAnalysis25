#!/usr/bin/env python3
import json
import torch
import logging
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_models():
    """
    Load your PyTorch models from .pth files.
    
    Returns:
        Ensemble model ready for inference
    """
    logger.info("Loading models from weights directory")
    
    weights_dir = Path(__file__).parent / 'weights'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for model_file in weights_dir.glob('*.pth'):
        try:
            # Replace with your actual model loading code
            # collect each model into a list of models
            
            logger.info(f"Loaded model: {model_file.name}")
        except Exception as e:
            logger.error(f"Error loading model {model_file.name}: {str(e)}")
    
    # TODO check need with ensemble method
    return None

def predict_style_changes(paragraphs):
    """
    Predict style changes between consecutive paragraphs using your model.
    
    Args:
        paragraphs: List of paragraphs (sentences) in the document
        model: Your loaded model
        
    Returns:
        List of binary predictions (0 or 1)
    """
    if len(paragraphs) <= 1:
        return []
    
    # Create pairs of consecutive sentences
    sentence_pairs = [(paragraphs[i], paragraphs[i+1]) for i in range(len(paragraphs)-1)]
    
    # TODO: Replace with your actual prediction logic
    # For now, return a placeholder
    predictions = [0] * len(sentence_pairs)
    
    return predictions

def run_with_models(problems, output_path):
    """
    Process problems and write predictions using your model.
    
    Args:
        problems: DataFrame of problem files
        output_path: Path to write solution files
        model: Your loaded model
    """
    logger.info(f'Processing {len(problems)} problems and writing outputs to {output_path}.')
    
    for _, item in problems.iterrows():
        # create output file path
        output_file = output_path / item["file"].replace("/problem-", "/solution-problem-").replace(".txt", ".json").replace("/train/", "/").replace("/test/", "/").replace("/validation/", "/")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # paragraphs is the list of sentences for a specific file
        paragraphs = item["paragraphs"]
        # TODO make predictions (copy paste ensembling code)
        predictions = predict_style_changes(paragraphs)
        
        # write predictions to output file
        with open(output_file, 'w') as out:
            prediction = {'changes': predictions}
            out.write(json.dumps(prediction))
            
        logger.info(f"Processed {Path(item['file']).name} successfully")
        
        # running models a lot, so clear cache frequently
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@click.command()
@click.option('--dataset', default='multi-author-writing-style-analysis-2025/multi-author-writing-spot-check-20250503-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))), help='The file where predictions should be written to.')
def main(dataset, output):
    """Main function that loads models, processes data, and writes predictions."""
    # Load your models
    # model = load_models()
    
    tira = Client()
    # Load the dataset
    input_df = tira.pd.inputs(dataset, formats=["multi-author-writing-style-analysis-problems"])
    logger.info(f"Successfully loaded dataset with {len(input_df)} problems")
    
    # process each difficulty subset
    for subtask in ["easy", "medium", "hard"]:
        subtask_problems = input_df[input_df["task"] == subtask]
        logger.info(f"Processing {len(subtask_problems)} problems for {subtask} subtask")
        run_with_models(subtask_problems, Path(output)) # TODO change this function
    
    logger.info("All problems processed successfully")

if __name__ == '__main__':
    main()