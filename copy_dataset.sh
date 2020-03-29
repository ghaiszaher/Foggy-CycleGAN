mkdir -p /content/dataset
for z in "${DRIVE_DATASETS}"*.zip
do
    printf "unzipping "
    printf $(basename "${z}")
    printf "...\n"
    unzip -q -n "${z}" -d /content/dataset/
    echo "Done."
done
echo "All files unzipped."
