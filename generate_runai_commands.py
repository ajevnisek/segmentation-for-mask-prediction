def generate_line(dataset, arch, encoder_name):
    return f"runai submit -g 1 --name {dataset}-{arch}-{encoder_name} -i  " \
    f"ajevnisek/semantic-segmentation-for-mask-prediction:{dataset}-{arch}-{encoder_name} --pvc=storage:/storage --large-shm"


for dataset in ['HAdobe5k', 'HCOCO']:
    for arch in ['PAN', 'DeepLabV3']:
        for encoder_name in ['resnet34', 'resnet101']:
            print(generate_line(dataset.lower(), arch.lower(), encoder_name.lower()))
