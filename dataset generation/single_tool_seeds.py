from typing import List
from dataclasses import dataclass


@dataclass
class Seed:
    instruction: str
    level: int
    pattern: str

class SingleToolSeeds:
    """12 diverse single-tool seeds using only apply and get operations"""

    @staticmethod
    def get_seeds() -> List[Seed]:
        return [
            # Level 1 - Full transformation name
            Seed(
                instruction="Transform '$path' using the $transformation_name transformation",
                level=1,
                pattern="apply"
            ),
            Seed(
                instruction="Show me the configuration settings for the $transformation_name transformation",
                level=1,
                pattern="get"
            ),
            # Level 2 - Source and target models mentioned
            Seed(
                instruction="Transform the $input_metamodel_name model $path into a $output_metamodel_name model",
                level=2,
                pattern="apply"
            ),
            Seed(
                instruction="Show me the configuration settings of the transformation that transforms a $input_metamodel_name into a $output_metamodel_name model",
                level=2,
                pattern="get"
                ),

            # Level 3 - Only target model mentioned
            Seed(
                instruction="Transform this model $path to $output_metamodel_name model",
                level=3,
                pattern="apply"
            ),
            Seed(
                instruction= "Show me the transformation that transforms this file $path to $output_metamodel_name model", 
                level=3,
                pattern="get"
            ),
        ]