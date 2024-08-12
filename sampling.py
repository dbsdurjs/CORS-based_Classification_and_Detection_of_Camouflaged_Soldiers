import os
from diffusers import AutoPipelineForText2Image
import torch
import matplotlib.pylab as plt

def sampling_func():
    num_images = 10
    num_model = 10

    prompt_list = ["In realistic, Camouflaged soldiers blend into a dense forest, hidden among the real trees and foliage.",
                    "In realistic, Military personnel in authentic camouflage gear move stealthily through a shadowy woodland.",
                    "In realistic, A squad of soldiers in full camouflage crouch behind thick underbrush in a genuine forest.",
                    "In realistic, Forest scene with soldiers in realistic camouflage uniforms, barely visible among the leaves.",
                    "In realistic, Heavily camouflaged soldiers stand motionless among the tall trees of an actual dense forest.",
                    "In realistic, Hidden soldiers in military camouflage gear silently navigate through dense, leafy vegetation.",
                    "In realistic, Camouflage-clad soldiers use natural foliage for cover in a sun-dappled forest setting.",
                    "In realistic, Soldiers in forest camouflage crouch and hide behind large tree trunks and bushes.",
                    "In realistic, Military unit in green and brown camouflage gear, blending perfectly with the real forest.",
                    "In realistic, Forest setting with soldiers in authentic camo gear, concealed by the natural environment.",
                    "In realistic, Soldiers in full camouflage gear lying low among thick forest vegetation, staying unseen.",
                    "In realistic, A group of camouflaged soldiers using the forest terrain to remain undetected.",
                    "In realistic, Camouflage-painted soldiers hidden behind bushes in a dense forest clearing.",
                    "In realistic, Soldiers in tactical camouflage gear, moving silently through the forest undergrowth.",
                    "In realistic, Stealthy soldiers in full camouflage, hidden in the shadows of a dense forest."
    ]

    for k in range(num_model):
        pipeline = AutoPipelineForText2Image.from_pretrained(f"yoon6173/result{k+1}", torch_dtype=torch.float16).to("cuda")
        print(f'{k+1}번째 model sampling')
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]        # a photo of sks camouflage soldier

            for j in range(num_images):
                image = pipeline(prompt, num_inference_steps=200, guidance_sacle=7.5).images[0]

                image_path = f'./sampling_camouflage_soldier/sample_{k + 1}_{(i * num_images) + (j + 1)}.png'
                plt.imshow(image)

                image.save(image_path)
                plt.axis('off')