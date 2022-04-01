# CycleGAN
***Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs.*** 
-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

**CycleGAN paper** [here](https://arxiv.org/abs/1703.10593)

### CycleGAN architecture
The model architecture is comprised of two generator models: one generator (Generator-A) for generating images for the first domain (Domain-A) and the second generator (Generator-B) for generating images for the second domain (Domain-B).
```
* Domain-B -> Generator-A -> Domain-A
* Domain-A -> Generator-B -> Domain-B
```
Each generator has a corresponding discriminator model (Discriminator-A and Discriminator-B). The discriminator model takes real images from Domain and generated images from Generator to predict whether they are real or fake.
```
* Domain-A -> Discriminator-A -> [Real/Fake]
* Domain-B -> Generator-A -> Discriminator-A -> [Real/Fake]
```
```
* Domain-B -> Discriminator-B -> [Real/Fake]
* Domain-A -> Generator-B -> Discriminator-B -> [Real/Fake]
```
![This is an image](https://github.com/AhmedAdelDraz/CycleGAN/blob/main/ouptuts/model_arch.png)
Figure 1. Overview of CycleGAN architecture

### Losses
- `Discriminator loss`: This ensures that the object class is modified while
training the model (as seen in the previous section).
- `Cycle loss`: The loss of recycling an image from the generated image to the
original to ensure that the surrounding pixels are not changed.
- `Identity loss`: The loss when an image of one class is passed through a
generator that is expected to convert an image of another class into the class
of the input image.
![This is an image](https://github.com/AhmedAdelDraz/CycleGAN/blob/main/ouptuts/mapping.png)

### Expected outputs
![This is an image](https://github.com/AhmedAdelDraz/CycleGAN/blob/main/ouptuts/outputs.png)
