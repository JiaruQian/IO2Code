"""
Command-line interface for DIOAgent
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from dio_agent.controller import DIOAgent
from dio_agent.config import Config, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DIOAgent - Evolutionary coding agent")

    parser.add_argument("initial_program", help="Path to the initial program file")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., dio_agent_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument(
        "--api-key-env",
        help="Environment variable name containing the LLM API key",
        default=None,
    )

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    return parser.parse_args()


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    api_key_override = None
    if args.api_key_env:
        api_key_override = os.environ.get(args.api_key_env)
        if not api_key_override:
            print(
                f"Error: Environment variable '{args.api_key_env}' is not set or empty"
            )
            return 1

        # Legacy task configs in this repo often hardcode ${OPENROUTER_API_KEY}.
        # Seed common env names before config loading so provider overrides can
        # still work without editing every task config.
        os.environ.setdefault("OPENROUTER_API_KEY", api_key_override)
        os.environ.setdefault("OPENAI_API_KEY", api_key_override)
        os.environ.setdefault("ANTHROPIC_API_KEY", api_key_override)

    is_anthropic_provider = args.api_key_env in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}

    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    # Load base config from file or defaults
    config = load_config(args.config)

    # Create config object with command-line overrides
    if args.api_base or args.api_key_env or args.primary_model or args.secondary_model:
        # Apply command-line overrides
        if args.api_base:
            config.llm.api_base = args.api_base
            print(f"Using API base: {config.llm.api_base}")

        if args.api_key_env:
            config.llm.api_key = api_key_override
            print(f"Using API key from env: {args.api_key_env}")

        if args.primary_model:
            config.llm.primary_model = args.primary_model
            print(f"Using primary model: {config.llm.primary_model}")

        if args.secondary_model:
            config.llm.secondary_model = args.secondary_model
            print(f"Using secondary model: {config.llm.secondary_model}")

        if is_anthropic_provider:
            # Anthropic-compatible chat endpoints in this repo should not receive a
            # non-null top_p from legacy task configs.
            config.llm.top_p = None
            print("Using Anthropic-compatible sampling params: top_p=None")

        # Rebuild models list to apply CLI overrides
        if args.primary_model or args.secondary_model:
            config.llm.rebuild_models()

        # Always propagate API overrides to already-instantiated model configs.
        model_overrides = {
            "api_base": config.llm.api_base,
            "api_key": config.llm.api_key,
        }
        if is_anthropic_provider:
            model_overrides["top_p"] = None
        config.llm.update_model_params(model_overrides, overwrite=True)

        if args.primary_model or args.secondary_model:
            print(f"Applied CLI model overrides - active models:")
            for i, model in enumerate(config.llm.models):
                print(f"  Model {i+1}: {model.name} (weight: {model.weight})")

    # Initialize DIOAgent
    try:
        dio_agent = DIOAgent(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            output_dir=args.output,
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            dio_agent.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {dio_agent.database.last_iteration})"
            )

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await dio_agent.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(dio_agent.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print(f"\nEvolution complete!")
        print(f"Best program metrics:")
        for name, value in best_program.metrics.items():
            # Handle mixed types: format numbers as floats, others as strings
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
