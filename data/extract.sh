sets=("train" "val")

for set in ${sets[@]}; do
	echo "extract imagenet/ILSVRC2012_img_${set}.tar into imagenet/${set}"
	mkdir imagenet/${set}
	tar -xf imagenet/ILSVRC2012_img_${set}.tar -C imagenet/${set}

	if [ ${set} = "train" ]; then
		tarfiles=imagenet/${set}/*.tar
		for tarfile in ${tarfiles[@]}; do
			echo "extract ${tarfile} to ${tarfile%.*}"
			mkdir ${tarfile%.*}
			tar -xf ${tarfile} -C ${tarfile%.*}
		done
	fi
done
