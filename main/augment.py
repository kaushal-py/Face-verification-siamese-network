import Augmentor
p = Augmentor.Pipeline("images/")

p.flip_left_right(probability=0.5)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1, max_factor=1.1)
p.skew_tilt(probability=0.3)

p.sample(10000)
