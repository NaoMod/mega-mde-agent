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
                instruction="Apply Mantis2XML to ./atl_zoo/SoftwareQualityControl2MantisBTFile/modelsExamples/MantisExample.xmi",
                level=1,
                pattern="apply"
            ),
            Seed(
                instruction="Show me the configuration settings for the PNML2XML transformation",
                level=1,
                pattern="get"
            ),
            # Level 2 - Source and target models mentioned
            Seed(
                instruction="Transform the KM3 model ./atl_zoo/KM32EMF/Sample-KM3.xmi into a DSL model",
                level=2,
                pattern="apply"
            ),
            Seed(
                instruction="Show me the configuration settings of the transformation that transforms a KM3 model into a DSL model",
                level=2,
                pattern="get"
                ),

            # Level 3 - Only target model mentioned
            Seed(
                instruction="Transform this model /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi to Relational model",
                level=3,
                pattern="apply"
            ),
            Seed(
                instruction= "Show me the transformation that transforms this file /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi to Relational model", 
                level=3,
                pattern="get"
            ),
        ]