#!/bin/bash

# Downloads and uncompresses all the required files for the Arabic dataset.
wget -O ar.zip https://www.dropbox.com/s/6wv7itqxwnxquxy/ar.zip?dl=1
unzip -o ar.zip
rm ar.zip
echo All required files for the Arabic dataset have been downloaded and un-compressed successfully.

# Downloads and uncompresses all the required files for the Esperanto dataset.
wget -O eo.zip https://www.dropbox.com/s/avhohdwyp3u2us9/eo.zip?dl=1
unzip -o eo.zip
rm eo.zip
echo All required files for the Esperanto dataset have been downloaded and un-compressed successfully.
