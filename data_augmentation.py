import Augmentor

# Setting the number of required augmented images
image_number = 755

# Creating a pipeline for augmentation by providing the path to original images
augmentation = Augmentor.Pipeline("C:/FYP/original dataset/Viral Pneumonia")

# Setting the augmentation with zooming, rotating and flipping
augmentation.zoom_random(probability=0.2, percentage_area=0.9)
augmentation.rotate(probability=1, max_left_rotation=0.14, max_right_rotation=0.14)
augmentation.flip_left_right(probability=0.4)

# Performing the augmentation
augmentation.status()
augmentation.sample(image_number)
