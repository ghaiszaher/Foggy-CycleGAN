mkdir -p $PROJECT_DIR/dataset
for z in "${DRIVE_DATASETS}"*.zip
do
    printf "unzipping "
    printf $(basename "${z}")
    printf "...\n"
    unzip -q "${z}" -d $PROJECT_DIR/dataset/
    echo "Done."
done
echo "All files unzipped."
