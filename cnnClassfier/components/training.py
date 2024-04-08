from cnnClassfier.entity.config_entity import TrainningConfig
from cnnClassfier.config.configuration import ConfigurationManager
import tensorflow as tf
from pathlib import Path








class Trainning:
    def __init__(self,config:TrainningConfig):
        self.config=config

    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    def train_valid_generator(self):
        datagenerator_kwargs=dict(
            rescale= 1./255,
            validation_split=.20)
        dataflow_kwargs=dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear")
        
        validation_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs)
        
        self.validation_generator=validation_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs

        )


        if self.config.params_is_augmentation:
            train_generator=tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2, 
                shear_range=0.2,
                zoom_range=0.2,
             **datagenerator_kwargs
             )
            
        else:
            train_generator=validation_datagenerator  
        
        self.train_generator=train_generator.flow_from_directory(

            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)



    def train(self,callback_list:list):
        self.steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps=self.validation_generator.samples // self.validation_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epoch,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=callback_list
            

            
        )
        self.save_model(path=self.config.trained_model_path, model=self.model)
