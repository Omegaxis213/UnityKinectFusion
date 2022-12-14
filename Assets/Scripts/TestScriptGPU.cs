using UnityEngine;
using Emgu.CV;
using System.IO;
struct TSDF
{
    public float tsdfValue;
    public float weight;
    public int color;
    public int colorWeight;
    public TSDF(float tsdfValue, float weight, int color, int colorWeight)
    {
        this.tsdfValue = tsdfValue;
        this.weight = weight;
        this.color = color;
        this.colorWeight = colorWeight;
    }
}

public class TestScriptGPU : MonoBehaviour
{
    [SerializeField]
    ComputeShader computeShader;
    Renderer rendererComponent;
    ComputeBuffer depthBuffer;
    ComputeBuffer leftDepthBuffer;
    ComputeBuffer normalBuffer;
    ComputeBuffer vertexBuffer;
    ComputeBuffer tsdfBuffer;
    ComputeBuffer smoothDepthBuffer;
    ComputeBuffer normalMapBuffer;
    ComputeBuffer vertexMapBuffer;
    ComputeBuffer ICPBuffer;
    ComputeBuffer ICPReductionBuffer;
    ComputeBuffer pointCloudBuffer;
    ComputeBuffer choleskyBuffer;
    ComputeBuffer cameraMatrixBuffer;
    ComputeBuffer invCameraMatrixBuffer;
    ComputeBuffer currentICPCameraMatrixBuffer;
    ComputeBuffer invCurrentICPCameraMatrixBuffer;
    static readonly int
        pixelBufferID = Shader.PropertyToID("pixelBuffer"),
        leftDepthBufferID = Shader.PropertyToID("leftDepthBuffer"),
        outputBufferID = Shader.PropertyToID("outputBuffer"),
        depthBufferID = Shader.PropertyToID("depthBuffer"),
        normalBufferID = Shader.PropertyToID("normalBuffer"),
        imageWidthID = Shader.PropertyToID("imageWidth"),
        imageHeightID = Shader.PropertyToID("imageHeight"),
        spatialWeightID = Shader.PropertyToID("spatialWeight"),
        rangeWeightID = Shader.PropertyToID("rangeWeight"),
        neighborSizeID = Shader.PropertyToID("neighborSize"),
        vertexBufferID = Shader.PropertyToID("vertexBuffer"),
        tsdfBufferID = Shader.PropertyToID("TSDFBuffer"),
        truncationDistID = Shader.PropertyToID("truncationDist"),
        voxelSizeID = Shader.PropertyToID("voxelSize"),
        roomSizeID = Shader.PropertyToID("roomSize"),
        cameraScaleID = Shader.PropertyToID("cameraScale"),
        colorIntrinsicMatrixID = Shader.PropertyToID("colorIntrinsicMatrix"),
        invColorIntrinsicMatrixID = Shader.PropertyToID("invColorIntrinsicMatrix"),
        rayTraceStepsID = Shader.PropertyToID("rayTraceSteps"),
        smoothDepthBufferID = Shader.PropertyToID("smoothDepthBuffer"),
        normalMapBufferID = Shader.PropertyToID("normalMapBuffer"),
        vertexMapBufferID = Shader.PropertyToID("vertexMapBuffer"),
        ICPBufferID = Shader.PropertyToID("ICPBuffer"),
        ICPReductionBufferID = Shader.PropertyToID("ICPReductionBuffer"),
        ICPThresholdDistanceID = Shader.PropertyToID("ICPThresholdDistance"),
        ICPThresholdRotationID = Shader.PropertyToID("ICPThresholdRotation"),
        pointCloudBufferID = Shader.PropertyToID("pointCloudBuffer"),
        maxTSDFWeightID = Shader.PropertyToID("maxTSDFWeight"),
        maxColorTSDFWeightID = Shader.PropertyToID("maxColorTSDFWeight"),
        colorIntrinsicMatrixOneID = Shader.PropertyToID("colorIntrinsicMatrixOne"),
        colorIntrinsicMatrixTwoID = Shader.PropertyToID("colorIntrinsicMatrixTwo"),
        invColorIntrinsicMatrixOneID = Shader.PropertyToID("invColorIntrinsicMatrixOne"),
        invColorIntrinsicMatrixTwoID = Shader.PropertyToID("invColorIntrinsicMatrixTwo"),
        splitID = Shader.PropertyToID("split"),
        reductionGroupSizeID = Shader.PropertyToID("reductionGroupSize"),
        CholeskyBufferID = Shader.PropertyToID("CholeskyBuffer"),
        cameraMatrixBufferID = Shader.PropertyToID("cameraMatrixBuffer"),
        invCameraMatrixBufferID = Shader.PropertyToID("invCameraMatrixBuffer"),
        currentICPCameraMatrixBufferID = Shader.PropertyToID("currentICPCameraMatrixBuffer"),
        invCurrentICPCameraMatrixBufferID = Shader.PropertyToID("invCurrentICPCameraMatrixBuffer");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    int[] defaultDepthArr;
    int FormatDepthBufferID;
    int DepthKernelID;
    int DrawDepthKernelID;
    int SmoothKernelID;
    int ComputeNormalsID;
    int TSDFUpdateID;
    int RenderTSDFID;
    int ICPKernelID;
    int ICPReductionKernelID;
    int ClearICPBufferID;
    int SolveCholeskyID;
    int UpdateCameraMatrixID;
    int SetCurrentCameraMatrixID;
    int imageWidth;
    int imageHeight;
    public float leftEyeTranslationDistance = 0f;
    public bool isPlayingRecording = true;
    public int renderMode = 1;

    public float spatialWeight = 75;
    public float rangeWeight = 75;
    public float truncationDist = 100f;
    public int neighborhoodSize = 10;
    public float roomSize = 5;
    public float cameraScale = 1;
    public int rayTraceSteps = 300;
    public float thresholdDistance = 5f;
    public float thresholdRotation = .1f;
    public int maxTSDFWeight = 64;
    public int maxColorTSDFWeight = 4;
    int voxelSize = 512;
    Matrix4x4 cameraMatrix;
    Matrix4x4 colorIntrinsicMatrix;
    Matrix4x4 colorIntrinsicMatrixOne;
    Matrix4x4 colorIntrinsicMatrixTwo;

    int frame = 0;
    bool isTracking;
    bool isFirst;
    public float pixelThreshold = 0;

    public int split = 0;

    int waveGroupSize = 256 * 64;
    StreamReader globalCameraMatrixReader;
    StreamReader CameraColorReader;
    string testDataPath = "./Assets/rgbd_dataset_freiburg1_xyz/";

    public float cameraXRot = 0;
    public float cameraYRot = 0;
    public float cameraZRot = 0;
    public float cameraXPos = 5;
    public float cameraYPos = 5;
    public float cameraZPos = 5;

    byte[] colors;
    ushort[] testArr;
    // Start is called before the first frame update
    void Start()
    {
        FormatDepthBufferID = computeShader.FindKernel("FormatDepthBuffer");
        DepthKernelID = computeShader.FindKernel("Depth");
        DrawDepthKernelID = computeShader.FindKernel("DrawDepth");
        SmoothKernelID = computeShader.FindKernel("Smooth");
        ComputeNormalsID = computeShader.FindKernel("ComputeNormals");
        TSDFUpdateID = computeShader.FindKernel("TSDFUpdate");
        RenderTSDFID = computeShader.FindKernel("RenderTSDF");
        ICPKernelID = computeShader.FindKernel("ICP");
        ICPReductionKernelID = computeShader.FindKernel("ICPReduction");
        ClearICPBufferID = computeShader.FindKernel("ClearICPBuffer");
        SolveCholeskyID = computeShader.FindKernel("SolveCholesky");
        UpdateCameraMatrixID = computeShader.FindKernel("UpdateCameraMatrix");
        SetCurrentCameraMatrixID = computeShader.FindKernel("SetCurrentCameraMatrix");
        Physics.autoSimulation = false;
        imageWidth = 640;
        imageHeight = 480;

        rendererComponent = GetComponent<Renderer>();
        rt = new RenderTexture(imageWidth, imageHeight, 0);
        rt.enableRandomWrite = true;
        outputTexture = new RenderTexture(imageWidth, imageHeight, 0);
        outputTexture.enableRandomWrite = true;
        RenderTexture.active = outputTexture;
        depthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        leftDepthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        smoothDepthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        normalMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        vertexMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        ICPBuffer = new ComputeBuffer(imageWidth * imageHeight * 32 / 64, 4);
        int reductionBufferSize = Mathf.CeilToInt((float)imageWidth * imageHeight / waveGroupSize / 2.0f);
        ICPReductionBuffer = new ComputeBuffer(reductionBufferSize * 32, 4);
        pointCloudBuffer = new ComputeBuffer(imageWidth * imageHeight * 3 / 2, 4);
        tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        blankBackground = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                blankBackground.SetPixel(j, i, Color.black);
            }
        }
        blankBackground.Apply();

        normalBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        vertexBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);

        tsdfBuffer = new ComputeBuffer(voxelSize * voxelSize * voxelSize, 16);

        choleskyBuffer = new ComputeBuffer(4 * 4, 4);
        cameraMatrixBuffer = new ComputeBuffer(1, 4 * 4 * 4);
        invCameraMatrixBuffer = new ComputeBuffer(1, 4 * 4 * 4);
        currentICPCameraMatrixBuffer = new ComputeBuffer(1, 4 * 4 * 4);
        invCurrentICPCameraMatrixBuffer = new ComputeBuffer(1, 4 * 4 * 4);
        computeShader.SetInt(imageHeightID, imageHeight);
        computeShader.SetInt(imageWidthID, imageWidth);
        computeShader.SetInt(voxelSizeID, voxelSize);
        computeShader.SetBuffer(FormatDepthBufferID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(FormatDepthBufferID, pointCloudBufferID, pointCloudBuffer);
        computeShader.SetBuffer(DepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetTexture(DrawDepthKernelID, pixelBufferID, rt);
        computeShader.SetTexture(DrawDepthKernelID, outputBufferID, outputTexture);
        computeShader.SetBuffer(SmoothKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(SmoothKernelID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(SmoothKernelID, smoothDepthBufferID, smoothDepthBuffer);
        computeShader.SetBuffer(ComputeNormalsID, normalBufferID, normalBuffer);
        computeShader.SetBuffer(ComputeNormalsID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(TSDFUpdateID, tsdfBufferID, tsdfBuffer);
        computeShader.SetBuffer(TSDFUpdateID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(TSDFUpdateID, vertexBufferID, vertexBuffer);
        computeShader.SetTexture(TSDFUpdateID, pixelBufferID, rt);
        computeShader.SetBuffer(RenderTSDFID, tsdfBufferID, tsdfBuffer);
        computeShader.SetBuffer(RenderTSDFID, normalMapBufferID, normalMapBuffer);
        computeShader.SetBuffer(RenderTSDFID, vertexMapBufferID, vertexMapBuffer);
        computeShader.SetBuffer(RenderTSDFID, normalBufferID, normalBuffer);
        computeShader.SetBuffer(RenderTSDFID, vertexBufferID, vertexBuffer);
        computeShader.SetTexture(RenderTSDFID, pixelBufferID, rt);
        computeShader.SetTexture(RenderTSDFID, outputBufferID, outputTexture);
        computeShader.SetBuffer(ClearICPBufferID, ICPBufferID, ICPBuffer);
        computeShader.SetBuffer(ICPKernelID, normalBufferID, normalBuffer);
        computeShader.SetBuffer(ICPKernelID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(ICPKernelID, normalMapBufferID, normalMapBuffer);
        computeShader.SetBuffer(ICPKernelID, vertexMapBufferID, vertexMapBuffer);
        computeShader.SetBuffer(ICPKernelID, ICPBufferID, ICPBuffer);
        computeShader.SetBuffer(ICPReductionKernelID, ICPBufferID, ICPBuffer);
        computeShader.SetBuffer(ICPReductionKernelID, ICPReductionBufferID, ICPReductionBuffer);
        computeShader.SetBuffer(SolveCholeskyID, CholeskyBufferID, choleskyBuffer);
        computeShader.SetBuffer(SolveCholeskyID, ICPReductionBufferID, ICPReductionBuffer);
        computeShader.SetBuffer(SetCurrentCameraMatrixID, CholeskyBufferID, choleskyBuffer);

        //Camera matrix buffer
        computeShader.SetBuffer(TSDFUpdateID, invCameraMatrixBufferID, invCameraMatrixBuffer);
        computeShader.SetBuffer(RenderTSDFID, cameraMatrixBufferID, cameraMatrixBuffer);
        computeShader.SetBuffer(ICPKernelID, invCameraMatrixBufferID, invCameraMatrixBuffer);
        computeShader.SetBuffer(ICPKernelID, currentICPCameraMatrixBufferID, currentICPCameraMatrixBuffer);
        computeShader.SetBuffer(SolveCholeskyID, currentICPCameraMatrixBufferID, currentICPCameraMatrixBuffer);
        computeShader.SetBuffer(SolveCholeskyID, invCurrentICPCameraMatrixBufferID, invCurrentICPCameraMatrixBuffer);
        computeShader.SetBuffer(UpdateCameraMatrixID, cameraMatrixBufferID, cameraMatrixBuffer);
        computeShader.SetBuffer(UpdateCameraMatrixID, invCameraMatrixBufferID, invCameraMatrixBuffer);
        computeShader.SetBuffer(UpdateCameraMatrixID, currentICPCameraMatrixBufferID, currentICPCameraMatrixBuffer);
        computeShader.SetBuffer(UpdateCameraMatrixID, invCurrentICPCameraMatrixBufferID, invCurrentICPCameraMatrixBuffer);
        computeShader.SetBuffer(SetCurrentCameraMatrixID, cameraMatrixBufferID, cameraMatrixBuffer);
        computeShader.SetBuffer(SetCurrentCameraMatrixID, invCameraMatrixBufferID, invCameraMatrixBuffer);
        computeShader.SetBuffer(SetCurrentCameraMatrixID, currentICPCameraMatrixBufferID, currentICPCameraMatrixBuffer);
        computeShader.SetBuffer(SetCurrentCameraMatrixID, invCurrentICPCameraMatrixBufferID, invCurrentICPCameraMatrixBuffer);
        defaultDepthArr = new int[imageHeight * imageWidth];
        Application.targetFrameRate = 60;
        cameraMatrix = new Matrix4x4(new Vector4(1, 0, 0, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(roomSize * .5f, roomSize * .5f, roomSize * .5f, 1));
        colorIntrinsicMatrix = new Matrix4x4(new Vector4(525f, 0, 0, 0), new Vector4(0, 525f, 0, 0), new Vector4(320, 240, 1, 0), new Vector4(0, 0, 0, 1));
        colorIntrinsicMatrixOne = new Matrix4x4(new Vector4(colorIntrinsicMatrix[0, 0] / 2, 0, 0, 0),
                                                new Vector4(0, colorIntrinsicMatrix[1, 1] / 2, 0, 0),
                                                new Vector4(colorIntrinsicMatrix[0, 2] / 2, colorIntrinsicMatrix[1, 2] / 2, 1, 0),
                                                new Vector4(0, 0, 0, 1));
        colorIntrinsicMatrixTwo = new Matrix4x4(new Vector4(colorIntrinsicMatrix[0, 0] / 4, 0, 0, 0),
                                                new Vector4(0, colorIntrinsicMatrix[1, 1] / 4, 0, 0),
                                                new Vector4(colorIntrinsicMatrix[0, 2] / 4, colorIntrinsicMatrix[1, 2] / 4, 1, 0),
                                                new Vector4(0, 0, 0, 1));
        isTracking = false;
        isFirst = false;
        frame = 0;
        isFirst = true;

        globalCameraMatrixReader = new StreamReader(testDataPath + "depth.txt");
        globalCameraMatrixReader.ReadLine();
        globalCameraMatrixReader.ReadLine();
        globalCameraMatrixReader.ReadLine();

        colors = new byte[640 * 480 * 3];
        testArr = new ushort[640 * 480];
        Matrix4x4[] cameraMatrixArr = new Matrix4x4[1];
        cameraMatrixArr[0] = cameraMatrix;
        Matrix4x4[] invCameraMatrixArr = new Matrix4x4[1];
        invCameraMatrixArr[0] = cameraMatrix.inverse;
        Matrix4x4[] cameraMatrixArrOne = new Matrix4x4[1];
        cameraMatrixArrOne[0] = cameraMatrix;
        Matrix4x4[] invCameraMatrixArrOne = new Matrix4x4[1];
        invCameraMatrixArrOne[0] = cameraMatrix.inverse;
        cameraMatrixBuffer.SetData(cameraMatrixArr);
        invCameraMatrixBuffer.SetData(invCameraMatrixArr);
        currentICPCameraMatrixBuffer.SetData(cameraMatrixArrOne);
        invCurrentICPCameraMatrixBuffer.SetData(invCameraMatrixArrOne);
    }

    private void OnEnable()
    {
        Start();
    }

    private void OnDisable()
    {
        depthBuffer.Release();
        leftDepthBuffer.Release();
        normalBuffer.Release();
        vertexBuffer.Release();
        tsdfBuffer.Release();
        ICPBuffer.Release();
        ICPReductionBuffer.Release();
        pointCloudBuffer.Release();
        outputTexture.Release();
        rt.Release();
        smoothDepthBuffer.Release();
        normalMapBuffer.Release();
        vertexMapBuffer.Release();
    }

    // Update is called once per frame
    void Update()
    {
        try
        {
            string[] tempArr = globalCameraMatrixReader.ReadLine().Split(' ');
            Mat testMat = CvInvoke.Imread(testDataPath + tempArr[1], Emgu.CV.CvEnum.ImreadModes.AnyDepth);
            tex.LoadRawTextureData(colors);
            tex.Apply();
            Graphics.Blit(tex, rt);

            testMat.CopyTo(testArr);
            for (int i = 0; i < testArr.Length; i++)
            {
                defaultDepthArr[i] = testArr[i] / 5;
            }
            depthBuffer.SetData(defaultDepthArr);
            KinectFusion();
        }
        catch (System.Exception)
        {
            cameraMatrix = new Matrix4x4(new Vector4(Mathf.Cos(cameraZRot) * Mathf.Cos(cameraYRot), Mathf.Sin(cameraZRot) * Mathf.Cos(cameraYRot), -Mathf.Sin(cameraYRot), 0),
                                         new Vector4(Mathf.Cos(cameraZRot) * Mathf.Sin(cameraYRot) * Mathf.Sin(cameraXRot) - Mathf.Sin(cameraZRot) * Mathf.Cos(cameraXRot), Mathf.Sin(cameraZRot) * Mathf.Sin(cameraYRot) * Mathf.Sin(cameraXRot) + Mathf.Cos(cameraZRot) * Mathf.Cos(cameraXRot), Mathf.Cos(cameraYRot) * Mathf.Sin(cameraXRot), 0),
                                         new Vector4(Mathf.Cos(cameraZRot) * Mathf.Sin(cameraYRot) * Mathf.Cos(cameraXRot) + Mathf.Sin(cameraZRot) * Mathf.Sin(cameraXRot), Mathf.Sin(cameraZRot) * Mathf.Sin(cameraYRot) * Mathf.Cos(cameraXRot) - Mathf.Cos(cameraZRot) * Mathf.Sin(cameraXRot), Mathf.Cos(cameraYRot) * Mathf.Cos(cameraXRot), 0), new Vector4(cameraXPos, cameraYPos, cameraZPos, 1));
            Matrix4x4[] cameraMatrixArr = new Matrix4x4[1];
            cameraMatrixArr[0] = cameraMatrix;
            Matrix4x4[] invCameraMatrixArr = new Matrix4x4[1];
            invCameraMatrixArr[0] = cameraMatrix.inverse;
            cameraMatrixBuffer.SetData(cameraMatrixArr);
            invCameraMatrixBuffer.SetData(invCameraMatrixArr);
            computeShader.SetInt(splitID, split);
            computeShader.Dispatch(RenderTSDFID, imageWidth / 8, imageHeight / 8, 1);
            
            rendererComponent.material.mainTexture = outputTexture;
        }
        
        
        frame++;
    }

    void KinectFusion()
    {
        computeShader.SetInt(maxTSDFWeightID, maxTSDFWeight);
        computeShader.SetInt(maxColorTSDFWeightID, maxColorTSDFWeight);
        computeShader.SetInt(splitID, split);
        computeShader.SetFloat(spatialWeightID, spatialWeight);
        computeShader.SetFloat(rangeWeightID, rangeWeight);
        computeShader.SetFloat(truncationDistID, truncationDist);
        computeShader.SetFloat(roomSizeID, roomSize);
        computeShader.SetFloat(cameraScaleID, cameraScale);
        if (neighborhoodSize < 0)
            neighborhoodSize = 0;
        computeShader.SetInt(neighborSizeID, neighborhoodSize);
        computeShader.SetInt(rayTraceStepsID, rayTraceSteps);
        computeShader.SetMatrix(colorIntrinsicMatrixID, colorIntrinsicMatrix);
        computeShader.SetMatrix(colorIntrinsicMatrixOneID, colorIntrinsicMatrixOne);
        computeShader.SetMatrix(colorIntrinsicMatrixTwoID, colorIntrinsicMatrixTwo);
        computeShader.SetMatrix(invColorIntrinsicMatrixID, colorIntrinsicMatrix.inverse);
        computeShader.SetMatrix(invColorIntrinsicMatrixOneID, colorIntrinsicMatrixOne.inverse);
        computeShader.SetMatrix(invColorIntrinsicMatrixTwoID, colorIntrinsicMatrixTwo.inverse);
        computeShader.Dispatch(SmoothKernelID, imageWidth / 8, imageHeight / 8, 1);
        //calculate normals at each point
        computeShader.Dispatch(ComputeNormalsID, imageWidth / 8, imageHeight / 8, 1);

        //ICP

        if (isTracking)
        {
            computeShader.SetFloat(ICPThresholdDistanceID, thresholdDistance);
            computeShader.SetFloat(ICPThresholdRotationID, Mathf.Cos(thresholdRotation));
            computeShader.Dispatch(SetCurrentCameraMatrixID, 1, 1, 1);
            for (int i = 0; i < 10; i++)
            {
                computeShader.Dispatch(ICPKernelID, imageWidth / 8, imageHeight / 8, 1);
                int reductionGroupSize = Mathf.CeilToInt((float)imageHeight * imageWidth / waveGroupSize / 2);
                computeShader.SetInt(reductionGroupSizeID, reductionGroupSize);
                computeShader.Dispatch(ICPReductionKernelID, reductionGroupSize, 1, 1);
                computeShader.Dispatch(SolveCholeskyID, 1, 1, 1);
            }
        }

        computeShader.Dispatch(UpdateCameraMatrixID, 1, 1, 1);

        //calculate TSDF
        computeShader.Dispatch(TSDFUpdateID, voxelSize / 8, voxelSize / 8, voxelSize / 8);
        //render TSDF
        computeShader.Dispatch(RenderTSDFID, imageWidth / 8, imageHeight / 8, 1);
        rendererComponent.material.mainTexture = outputTexture;
        isTracking = true;
    }
}
