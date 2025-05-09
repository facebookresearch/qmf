#!/usr/bin/fish
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#
# convert  raw/boz2.ppm -font helvetica -pointsize 100 \
#     -draw " fill rgba(0, 0, 0, 0.1) text 167,554 'userName'" \
#     -pointsize 70 -draw " fill white gravity center text 550,810 \"Our method (QMF)\""  \
#     -pointsize 70 -draw " fill white gravity center text 550,890 \"size: 346.2 KB\""  \
#     -pointsize 70 -draw " fill white gravity center text 550,970 \"wall time: 400.6 us\""  \
#     -pointsize 70 -draw " fill white gravity center text -550,810 \"Sparse BS\""  \
#     -pointsize 70 -draw " fill white gravity center text -550,890 \"size: 24.4 MB\""  \
#     -pointsize 70 -draw " fill white gravity center text -550,970 \"wall time: 2.1 ms --\"" \
#     /home/romanfedotov/Documents/paperSF/video/images/diffScale.tga -compose over -composite \
#     boz2.ppm

convert  raw/boz3.ppm  \
    -draw " fill rgba(0, 0, 0, 0.1) text 167,554 'userName'" \
    -pointsize 70 -draw " fill white gravity center text -990,810 \"Sparse BS\""  \
    -pointsize 70 -draw " fill white gravity center text 60,810 \"CSFB\""  \
    -pointsize 70 -draw " fill white gravity center text 1100,810 \"CSFB + new loss\""  \
    /home/romanfedotov/Documents/paperSF/video/images/diffScale3.tga -compose over -composite \
    boz3.ppm
# end


    # -pointsize 100 -draw " fill rgba(255, 0, 0, 0.9) gravity center text 530,900 'QMF-Blend'"  \
      #
# -font arial -gravity northwest \
#   -annotate +78+167 "my first text" \
#   -annotate +85+213 "center text" \
#   -fill none -stroke red -strokewidth 10 \
#   -draw "rectangle 65,155 275,263" -alpha off \


# title crator
convert -size 3840x2160 xc:black \
  -fill wite -pointsize 36 \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-200 'QMF-Blend algorithm'" \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-90 'Bowen model'" \
  QMFBoz.png

convert -size 3840x2160 xc:black \
  -fill wite -pointsize 36 \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-200 'QMF-Blend algorithm'" \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-90 'Jupiter model'" \
  QMFJupiter.png

convert -size 3840x2160 xc:black \
  -fill wite -pointsize 36 \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-200 'QMF-Blend algorithm'" \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-90 'Carlos model'" \
  QMFCarlos.png

convert -size 3840x2160 xc:black \
  -fill wite -pointsize 36 \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-200 'CSFB + new loss'" \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-90 'Bowen model'" \
  CSFBBoz.png

convert -size 3840x2160 xc:black \
  -fill wite -pointsize 36 \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-200 'CSFB + new loss'" \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-90 'Jupiter model'" \
  CSFBJupiter.png

convert -size 3840x2160 xc:black \
  -fill wite -pointsize 36 \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-200 'CSFB + new loss'" \
  -pointsize 100 -draw " fill rgba(255, 255, 255, 1.0) gravity center text 0,-90 'Carlos model'" \
  CSFBCarlos.png
