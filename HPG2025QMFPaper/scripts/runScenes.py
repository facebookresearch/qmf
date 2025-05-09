#!/usr/bin/python
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

# import subprocess

abcFolder = "/home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/"
outFolder = "/home/romanfedotov/Documents/paperSF/runtimeData/"
names = (
    # "Aura_3Faces_float.abc",
    # "Boz_3Faces_float.abc",
    # "Jupiter_3Faces_float.abc",
    # "Proteus_3Faces_float.abc",
    # "Louise_3Faces_float.abc",
    # "Carlos_3Faces_float.abc",
    # "Aura_3Faces_Q8.abc",
    # "Boz_3Faces_Q8.abc",
    # "Jupiter_3Faces_Q8.abc",
    # "Proteus_3Faces_Q8.abc",
    # "Louise_3Faces_Q8.abc",
    # "Carlos_3Faces_Q8.abc",
    # -----
    # "Aura_3Faces_Q8.abc",
    # "Boz_3Faces_Q11_11_10.abc",
    # "Jupiter_3Faces_Q8.abc",
    # "Proteus_3Faces_Q11_11_10.abc",
    # "Louise_3Faces_Q11_11_10.abc",
    "Carlos_3Faces_Q11_11_10.abc",
)

curDir = os.getcwd()
os.chdir("/home/romanfedotov/fbsource/arvr/projects/dgdt")

for i in names:
    os.system("rm ~/temp/sparseFactorizationDemo/*.dgdt")
    os.system(
        f"buck run @arvr/mode/linux/cuda12/dev //arvr/projects/dgdt/dgdt/tools:abc_to_dgdt -- \
          --output_dir ~/temp/sparseFactorizationDemo/ \
          --graph_description_file_name  \
          ~/fbsource/arvr/projects/dgdt/dgdt/tools/optix_viewer/facebook/test_models/blendshapes_skinDec_sparseF_3faces.json \
          ~/Documents/houdini/projects/blendshapeAndSkin/abc/{i}"
    )
    os.system(
        f"buck run @arvr/mode/linux/cuda12/opt //arvr/projects/dgdt/dgdt/tools/embree_viewer:embree_viewer -- \
              ~/temp/sparseFactorizationDemo/ --times_thru_seq=1  --csv_output {outFolder}{i}.csv"
    )

os.chdir(curDir)


# $ buck run @arvr/mode/linux/cuda12/opt //arvr/projects/dgdt/dgdt/tools/embree_viewer:embree_viewer -- \
#               ~/temp/sparseFactorizationDemo/ --times_thru_seq=1  --camera "20 0.0 100.0 20.0 0.0 0.0"

# aura
# buck run @arvr/mode/linux/cuda12/opt //arvr/projects/dgdt/dgdt/tools/embree_viewer:embree_headless -- \
#               ~/temp/sparseFactorizationDemo/ --times_thru_seq=1 --output-dir=/home/romanfedotov/Documents/paperSF/video --camera "20 -3 100  20 -3 0"


# $ ffmpeg -framerate 25 -i 000_frame%3d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4

# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Aura_3Faces_float.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Aura_3Faces_Q8.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Boz_3Faces_float.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Boz_3Faces_Q8.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Jupiter_3Faces_float.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Jupiter_3Faces_Q8.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Proteus_3Faces_float.abc
# /home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/Proteus_3Faces_Q8.abc
