#!/usr/bin/python
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import glob
import os

# import subprocess

abcFolder = "/home/romanfedotov/Documents/houdini/projects/blendshapeAndSkin/abc/"
# outFolder = "/home/romanfedotov/Documents/paperSF/runtimeData/"
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
    #
    # "Aura_3Faces_Q16.abc",
    # "Boz_3Faces_Q16.abc",
    "Jupiter_3Faces_Q16.abc",
    # "Proteus_3Faces_Q16.abc",
    # "Louise_3Faces_Q16.abc",
    # "Carlos_3Faces_Q16.abc",
)
jsons = {
    "2face": "blendshapes_sparseF_2faces.json",
    "2faceWhite": "blendshapes_sparseF_2facesWhite.json",
    "CSFB_3face": "blendshapes_CSFB_3faces.json",
}
camParams = {
    "carlos": "10 -3 110  10 -3 0",
    "boz": "10 -5 110  10 -5 0",
    "jupiter": "10 -7 110  10 -7 0",
}

videos = (
    # ("boz", "Boz_3Faces_CSFB.abc", "CSFB_3face", "20 -5 110  20 -5 0"),
    # (
    #     "jupiter",
    #     "Jupiter_3Faces_Q8.abc",
    #     "CSFB_3face",
    #     "20 -7 110  20 -7 0",
    # ),
    # (
    #     "carlos",
    #     "Carlos_3Faces_CSFB.abc",
    #     "CSFB_3face",
    #     "20 -3 110  20 -3 0",
    # ),
    # -----------
    (
        "jupiter",
        "Jupiter_3Faces_Q16.abc",
        "2face",
        "10 -7 110  10 -7 0",
    ),
    (
        "jupiter",
        "Jupiter_3Faces_Q16.abc",
        "2faceWhite",
        "10 -7 110  10 -7 0",
    ),
    # ("boz", "Boz_3Faces_Q16.abc", "2face", "10 -5 110  10 -5 0"),
    # ("boz", "Boz_3Faces_Q16.abc", "2faceWhite", "10 -5 110  10 -5 0"),
    # (
    #     "carlos",
    #     "Carlos_3Faces_Q16.abc",
    #     "2face",
    #     "10 -3 110  10 -3 0",
    # ),
    # (
    #     "carlos",
    #     "Carlos_3Faces_Q16.abc",
    #     "2faceWhite",
    #     "10 -3 110  10 -3 0",
    # ),
)
if (
    False
):  # -------------------------------generate raw files-------------------------------
    curDir = os.getcwd()
    os.chdir("/home/romanfedotov/fbsource/arvr/projects/dgdt")

    for model, abcFile, experiment, camParam in videos:
        folderName = (
            f"/home/romanfedotov/Documents/paperSF/video/raw/{model}-{experiment}/"
        )
        if not os.path.exists(folderName):
            os.makedirs(folderName)

        os.system("rm ~/temp/sparseFactorizationDemo/*.dgdt")
        os.system(f"rm {folderName}/*.ppm")
        s = f"buck run @arvr/mode/linux/cuda12/dev //arvr/projects/dgdt/dgdt/tools:abc_to_dgdt -- \
              --output_dir ~/temp/sparseFactorizationDemo/ \
              --graph_description_file_name  \
              ~/fbsource/arvr/projects/dgdt/dgdt/tools/optix_viewer/facebook/test_models/{jsons[experiment]} \
              ~/Documents/houdini/projects/blendshapeAndSkin/abc/{abcFile}"
        print(s)
        print("---------")
        os.system(s)
        os.system(
            f'buck run @arvr/mode/linux/cuda12/opt //arvr/projects/dgdt/dgdt/tools/embree_viewer:embree_headless -- \
                  ~/temp/sparseFactorizationDemo/ --times_thru_seq=1 --camera "{camParam}" -camera_fov 19 -image_width 3840  -image_height 2160\
                  --output-dir={folderName}'
        )
    os.chdir(curDir)
# -image_height (Height of the rendering image) type: uint32 default: 720
#     -image_width

# -----------------------------------Annotation-----------------------------------
annotations = (
    # model,     modelSize,   cpuTime,     bsSize,     bsTime
    ("jupiter", "65.4 KB", "88.9 us", "6.4 MB", "217.0 us"),
    ("boz", "346.2 KB", "415.8 us", "24.4 MB", "925.6 us"),
    ("carlos", "434.2 KB", "500.9 us", "25.0 MB", "3.0 ms"),
)
if True:
    for experiment in ("2faceWhite",):  # "2faceWhite",
        for model, modelSize, cpuTime, bsSize, bsTime in annotations:
            inFolderName = (
                f"/home/romanfedotov/Documents/paperSF/video/raw/{model}-{experiment}/"
            )
            outFolderName = f"/home/romanfedotov/Documents/paperSF/video/annotated/{model}-{experiment}/"
            if not os.path.exists(outFolderName):
                os.makedirs(outFolderName)

            os.system(f"rm {outFolderName}/*.ppm")
            if experiment == "2face":
                files = glob.glob("*.ppm", root_dir=inFolderName)
                for f in files:
                    os.system(
                        f"convert  {inFolderName}{f} /home/romanfedotov/Documents/paperSF/video/images/diffScale.tga -compose over -composite {outFolderName}{f}"
                    )
            else:
                os.system(f"cp {inFolderName}*.ppm {outFolderName}")

            os.system(
                f'mogrify -font helvetica -pointsize 100 \
                -pointsize 70 -draw " fill white gravity center text 550,810 \\"Our method (QMF)\\""  \
                -pointsize 70 -draw " fill white gravity center text 550,890 \\"size: {modelSize}\\""  \
                -pointsize 70 -draw " fill white gravity center text 550,970  \\"wall time: {cpuTime}\\""  \
                -pointsize 70 -draw " fill white gravity center text -550,810 \\"Sparse BS\\""  \
                -pointsize 70 -draw " fill white gravity center text -550,890 \\"size: {bsSize}\\""  \
                -pointsize 70 -draw " fill white gravity center text -550,970 \\"wall time: {bsTime}\\"" \
                {outFolderName}*.ppm'
            )
if False:
    for model in ["jupiter", "carlos"]:  # "boz",
        experiment = "CSFB_3face"
        inFolderName = (
            f"/home/romanfedotov/Documents/paperSF/video/raw/{model}-{experiment}/"
        )
        outFolderName = f"/home/romanfedotov/Documents/paperSF/video/annotated/{model}-{experiment}/"

        if not os.path.exists(outFolderName):
            os.makedirs(outFolderName)
        os.system(f"rm {outFolderName}/*.ppm")
        files = glob.glob("*.ppm", root_dir=inFolderName)
        for f in files:
            os.system(
                f"convert  {inFolderName}{f} /home/romanfedotov/Documents/paperSF/video/images/diffScale3.tga -compose over -composite {outFolderName}{f}"
            )

        os.system(
            f'mogrify -font helvetica -pointsize 100 \
                -pointsize 70 -draw " fill white gravity center text -990,810 \\"Sparse BS\\""  \
                -pointsize 70 -draw " fill white gravity center text 60,810 \\"CSFB\\""  \
                -pointsize 70 -draw " fill white gravity center text 1100,810 \\"CSFB + new loss\\""  \
                {outFolderName}*.ppm'
        )


# carlos   --camera "10 -3 110  10 -3 0"
# boz      --camera "10 -5 110  10 -5 0"
# jupiter  --camera "10 -7 110  10 -7 0"

# $ buck run @arvr/mode/linux/cuda12/opt //arvr/projects/dgdt/dgdt/tools/embree_viewer:embree_viewer -- \
#               ~/temp/sparseFactorizationDemo/ --times_thru_seq=1  --camera "20 0.0 100.0 20.0 0.0 0.0"

# aura 3
# buck run @arvr/mode/linux/cuda12/opt //arvr/projects/dgdt/dgdt/tools/embree_viewer:embree_headless -- \
#               ~/temp/sparseFactorizationDemo/ --times_thru_seq=1 --output-dir=/home/romanfedotov/Documents/paperSF/video --camera "20 -3 100  20 -3 0"


# $ ffmpeg -framerate 25 -i 000_frame%3d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4
# lossless
# $ ffmpeg -framerate 25 -i frame%03d.ppm  -c:v libx264 -crf 0 outputB.mp4
