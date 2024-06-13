#!/usr/bin/env nextflow

params.root_dir = '/cephyr/users/jichenl/Alvis/work/PiNN/arch_bs_clp_lr'
params.max_step = 2000000
params.batch_size = 64

process checkFolders {
    input:
    val root_dir
    val max_step

    output:
    path folders_to_process

    script:
    def folders_to_process = []
    
    // 遍历根目录下的文件夹
    new File(root_dir).eachDirRecurse { dir ->
        def trainFile = new File(dir, 'train.tfr')
        
        if (trainFile.exists()) {
            def modelFilePattern = ~/model.ckpt-(\d+)\.meta/
            
            dir.eachFileRecurse { file ->
                def matcher = file.name =~ modelFilePattern
                if (matcher) {
                    def match = matcher[0]
                    def step = match[1].toInteger()
                    println "Found file: ${file.absolutePath} with step: ${step}"
                    if (step < max_step) {
                        folders_to_process << file.parentFile
                        return
                    }
                }
            }
        }
    }

    def output_file = file('folders_to_process.txt')
    output_file.text = folders_to_process.join('\n')
    output_file.copyTo(file('folders_to_process.txt'))
}

process executeScript {
    input:
    val path

    script:

    def model_dir = new File(path)

    """
    pinn train ${model_dir}/params.yml -d ${model_dir} --train-ds ${model_dir.parentFile}/train.yml --eval-ds ${model_dir.parentFile}/eval.yml -b ${params.batch_size} --train-steps ${params.max_step}
    """
}

workflow {
    // restart_channel = Channel.of()
    // checkFolders(params.root_dir, params.max_step)
    files = Channel.fromList(file('/cephyr/users/jichenl/Alvis/work/PiNN/restart.txt').readLines())
    | executeScript
}