import imageprocessor

orig = open_image_from_file('original.jpg')
gray = gray_image(orig)
scaled = scale_down_image(orig, (320, 200))
gray_then_scale = scale_down_image(gray_image(orig), (320, 200))
scale_then_gray = gray_image(scale_down_image(orig, (320, 200)))
gray.save("gray.jpg")
scaled.save("scaled.jpg")
gray_then_scale.save("gray_then_scale.jpg")
scale_then_gray.save("scale_then_gray.jpg")
