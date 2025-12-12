# QMF-Blend: Quantized Matrix Factorization for Efficient Blendshape Compression

**Meta Reality Labs Research**

  Roman Fedotov, Brian Budge, Ladislav Kavan,


## How to use script

0. Install python libraries:
    ```bash
    pip3 install libigl torch numpy scipy
    ```
1. Download the models located in the `in` folder from the repository `https://github.com/facebookresearch/compskin`  to sub folder `models`

2. To run script edit  `runModel()` function
    change list of the models in the main loop:
    ```python
    for model in ["jupiter", "aura"]: # here we optimze jupiter and aura models
    ```
    change parameters of optimization:
    ```python
        p = Params(
            model=model,
            numIterations=20_000,
            numColumnB=200,
            numNz=posterParams[model],
            power=2,
            alpha=modelAlpas[model],
            lr=1e-2,
            seed=1,
            initB=1e-3,
            initC=1e-3,
            numBits=16, # quantize with 16 bits
        )
    ```
    and run script
    ```
    $ python3 sparseFactorization.py
    ```
3. Script will create `out` folder and will use this as output folder. (it will create
    subfolders and file names automatically based on parameters of optimization
    For example one optimizatio of model will generatet quantized and float point output data:

    ```
    data - compressed data file in numpy format npz if quantized has suffix _Q<numBits>
    objBS - all blendshapes in obj format
    objAn - testing animation in obj format
    the name of the folder encoded several paramters of optimization:
        Nnz110000 - 110'000 non zeros values
        CM - wrinkles map was on
        Nb200 - 200 number of columns in matrix B (row in C)
        a25 - alpha 25.0

    Example:

    📁aura
    └── 📁Nnz35000_CM_Nb200_a9
        ├── 📁objAn
        ├── 📁objAn_Q111110
        ├── 📁objAn_Q16
        ├── 📁objAn_Q8
        ├── 📁objBS
        ├── 📁objBS_Q111110
        ├── 📁objBS_Q16
        ├── 📁objBS_Q8
        ├── data.npz
        ├── data_Q111110.npz
        ├── data_Q16.npz
        ├── data_Q8.npz
        └── numNz.npz

    ```
4. To compare with blendshapes without compression uncomment lines in `main()` function
    ```python
    # for model in ("aura", "bowen", "jupiter", "proteus"):
    #     outputBlendshapesObj(model, calcGeo(model))
    ```
    it will generate folder with original blendshapes
    ```
    📁bowen
    └── objBS_lossless
    ```
## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE).

