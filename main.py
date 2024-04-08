from cnnClassfier import logger
from cnnClassfier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassfier.pipeline.stage_02_prepare_base_model import PrepaprebaseModelTrainingPipeline
from cnnClassfier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassfier.pipeline.stage_04_evaluation import EvaluationPipeline



STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} Started <<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME}  completed  <<<<<")

except Exception as e:
    logger.exception(e)
    raise e  



STAGE_NAME="Prepare Base Model"
try:
    logger.info(f"***********************")
    logger.info(f">>>> stage {STAGE_NAME} Started <<<<<")
    prepare_base_model=PrepaprebaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>> stage {STAGE_NAME}  completed  <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME= "Training"
if __name__ == '__main__':
  try:
    logger.info(f"***********************")
    logger.info(f">>>> stage {STAGE_NAME} Started <<<<<")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME}  completed  <<<<<")

  except Exception as e:
    logger.exception(e)
    raise e  




STAGE_NAME= "Evaluation"

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