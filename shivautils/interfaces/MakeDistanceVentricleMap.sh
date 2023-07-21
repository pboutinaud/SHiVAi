#!/bin/bash

# Script to run niimath to create Distance Ventricle Maps for postprocessing of WMH predictions

im=$1
out=$2

niimath $im -binv -edt $out


