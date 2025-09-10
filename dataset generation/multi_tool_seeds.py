from typing import List
from dataclasses import dataclass


@dataclass
class Seed:
    instruction: str
    level: int
    pattern: str

class MultiToolSeeds:
    """36 multi-tool seeds combining get and apply operations"""

    @staticmethod
    def get_seeds() -> List[Seed]:
        return [
            # LEVEL 1 - Full transformation names 
            Seed(
                instruction="First show me the configuration of $transformation_name1 and then transform $path2 using $transformation_name2 transformation",
                level=1,
                pattern="get, apply"
            ),
            Seed(
                instruction="Compare the configurations of the $transformation_name1 and $transformation_name2 transformations",
                level=1,
                pattern="get, get"
            ),
            Seed(
                instruction="Transform $path1 with $transformation_name1 and then check the $transformation_name2 configuration ",
                level=1,
                pattern="apply, get"
            ),
            Seed(
                instruction="Transform $path1 using $transformation_name1 and then transform $path2 using $transformation_name2",
                level=1,
                pattern="apply, apply"
            ),
            # LEVEL 2 - Source and target models mentioned
            Seed(
                instruction="Show me the configuration settings of the transformation that transforms a $input_metamodel_name1 into a $output_metamodel_name1 model and then transform $path2 from this model $input_metamodel_name2 to a $output_metamodel_name2 model",
                level=2,
                pattern="get, apply"

            ),
            Seed(
                instruction="Transform the $input_metamodel_name1 model $path1 to $output_metamodel_name1 and then compare the configurations of the of the transformation that transforms a $input_metamodel_name2 into a $output_metamodel_name2 ",
                level=2,
                pattern="apply, get"
            ),
            Seed(
                instruction="Transform the $input_metamodel_name1 model $path1 to $output_metamodel_name1 and then transform the $input_metamodel_name2 model $path2 to $output_metamodel_name2",
                level=2,
                pattern="apply, apply"
            ),
            Seed(
                instruction="Show me the configuration settings of the transformation that transforms a $input_metamodel_name1 into a $output_metamodel_name1 and then show me the configuration settings of the transformation that transforms a $input_metamodel_name2 into a $output_metamodel_name2",
                level=2,
                pattern="get, get"
            ),

            # LEVEL 3 - Only target model mentioned 
            Seed(
                instruction="Transform this model $path1 to $output_metamodel_name1 model and then transform this model $path2 to $output_metamodel_name2 model",
                level=3,
                pattern="apply, apply"
            ),
            Seed(
                instruction= "Show me the transformation that transforms this file $path1 to $output_metamodel_name1 model then transform this model $path2 to $output_metamodel_name2 model", 
                level=3,
                pattern="get, apply"
            ),
            Seed(
                instruction= "Show me the transformation that transforms this file $path1 to $output_metamodel_name1 model then show me the transformation that transforms this file $path2 to to $output_metamodel_name2", 
                level=3,
                pattern="get, get"
            ),
            Seed(
                instruction= "Transform this model $path1 to $output_metamodel_name1 model then show me the transformation that transforms this file $path2 to to $output_metamodel_name2 model ", 
                level=3,
                pattern="apply, get"
            )
            ]
    