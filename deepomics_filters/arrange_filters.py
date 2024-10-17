from PIL import Image

image_files = [f'deepomics LAYER 2 filter {i}.png' for i in range(0, 64)]

images = [Image.open(img) for img in image_files]

img_width, img_height = images[0].size

grid_width = img_width * 8
grid_height = img_height * 8
grid_image = Image.new('RGB', (grid_width, grid_height))

for i, img in enumerate(images):
    row = i // 8  
    col = i % 8   
    grid_image.paste(img, (col * img_width, row * img_height))
    
grid_image.save('stitched_sequence_logos.jpg')

