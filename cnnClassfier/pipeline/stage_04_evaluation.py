import tensorflow as tf 
from pathlib import Path
from cnnClassfier.config.configuration import ConfigurationManager
from cnnClassfier.components.evaluation import Evaluation
from cnnClassfier import logger


STAGE_NAME= "Evaluation"

class EvaluationPipeline:
    def __init__(self) :
        pass
    def main(self):
        config= ConfigurationManager()
        val_config=config.get_validation_config()
        evaulation=Evaluation(val_config)
        evaulation.evaluation()
        evaulation.save_score()




if __name__ == '__main__':
  try:
    logger.info(f"***********************")
    logger.info(f">>>> stage {STAGE_NAME} Started <<<<<")
    obj=EvaluationPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME}  completed  <<<<<")

  except Exception as e:
    logger.exception(e)
    raise e  