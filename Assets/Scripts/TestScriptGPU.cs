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
    [SerializeField]
    ComputeShader octreeShader;
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
        invCurrentICPCameraMatrixBufferID = Shader.PropertyToID("invCurrentICPCameraMatrixBuffer"),
        tailBufferID = Shader.PropertyToID("tail"),
        offsetBufferID = Shader.PropertyToID("offset"),
        idChildArrBufferID = Shader.PropertyToID("idChildArr"),
        xyzKeyBufferID = Shader.PropertyToID("xyzKey"),
        splitFlagBufferID = Shader.PropertyToID("splitFlag"),
        currentLayerID = Shader.PropertyToID("currentLayer"),
        branchLayerID = Shader.PropertyToID("branchLayer"),
        treeDepthID = Shader.PropertyToID("treeDepth"),
        truncationDistID = Shader.PropertyToID("truncationDist"),
        maxSizeID = Shader.PropertyToID("maxSize"),
        resultBufferID = Shader.PropertyToID("resultBuffer"),
        sdfBufferID = Shader.PropertyToID("sdfBuffer"),
        weightBufferID = Shader.PropertyToID("weightBuffer");
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
    public float truncationDist = .01f;
    public int neighborhoodSize = 10;
    public float roomSize = 8;
    public float cameraScale = 1;
    public int rayTraceSteps = 300;
    public float thresholdDistance = .02f;
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

    Vector3[] normalArr;
    Vector3[] vertexArr;

    int treeDepth = 10;
    double bufferSizeConstant = .03;
    int branchLayer = 7;
    int[] idChildArr;
    int[] xyzKey;
    int[] tail;
    int[] offset;
    int[] splitFlag;
    float[] dataLayerSDF;
    float[] dataLayerWeights;
    int SplitNodesFlagKernelID;
    int SplitNodesScanKernelID;
    int SplitNodesPropKernelID;
    int SplitNodesTailKernelID;
    int updateTopLayerKernelID;
    int updateSDFLayerKernelID;
    int surfacePredictKernelID;
    int DebugFunctionKernelID;
    ComputeBuffer tailBuffer;
    ComputeBuffer offsetBuffer;
    ComputeBuffer idChildArrBuffer;
    ComputeBuffer xyzKeyBuffer;
    ComputeBuffer splitFlagBuffer;
    ComputeBuffer resultBuffer;
    ComputeBuffer sdfBuffer;
    ComputeBuffer weightBuffer;
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
        normalArr = new Vector3[imageWidth * imageHeight];
        vertexArr = new Vector3[imageWidth * imageHeight];

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
        colorIntrinsicMatrix = new Matrix4x4(new Vector4(320, 0, 0, 0), new Vector4(0, 240f, 0, 0), new Vector4(320, 240, 1, 0), new Vector4(0, 0, 0, 1));
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

        SplitNodesFlagKernelID = octreeShader.FindKernel("SplitNodesFlag");
        SplitNodesScanKernelID = octreeShader.FindKernel("SplitNodesScan");
        SplitNodesPropKernelID = octreeShader.FindKernel("SplitNodesProp");
        SplitNodesTailKernelID = octreeShader.FindKernel("SplitNodesTail");
        updateTopLayerKernelID = octreeShader.FindKernel("UpdateTopLayer");
        updateSDFLayerKernelID = octreeShader.FindKernel("UpdateSDFLayer");
        surfacePredictKernelID = octreeShader.FindKernel("SurfacePredict");
        DebugFunctionKernelID = octreeShader.FindKernel("DebugFunction");
        resultBuffer = new ComputeBuffer(1, 16);
        tailBuffer = new ComputeBuffer(treeDepth + 1, 4);
        offsetBuffer = new ComputeBuffer(treeDepth + 2, 4);

        int maxSize = 0;
        offset = new int[treeDepth + 2];
        for (int i = 0; i <= treeDepth; i++)
        {
            if (i <= branchLayer)
            {
                offset[i + 1] = 1 << i * 3;
            }
            else
            {
                offset[i + 1] = (int)(bufferSizeConstant * (1 << i * 3));
            }
            maxSize = Mathf.Max(maxSize, offset[i + 1]);
            offset[i + 1] += offset[i];
        }
        splitFlagBuffer = new ComputeBuffer(maxSize + 1, 4);
        idChildArrBuffer = new ComputeBuffer(offset[treeDepth], 4);
        xyzKeyBuffer = new ComputeBuffer(offset[treeDepth + 1], 4);
        idChildArr = new int[offset[treeDepth]];
        xyzKey = new int[offset[treeDepth + 1]];
        tail = new int[treeDepth + 1];
        splitFlag = new int[maxSize + 1];
        // initialize top layers
        for (int i = 0; i < branchLayer; i++)
        {
            tail[i] = 1 << (i * 3);
        }
        // initialize branch layer
        tail[branchLayer] = 1 << (branchLayer * 3);
        for (int i = 0; i < idChildArr.Length; i++)
        {
            idChildArr[i] = -1;
        }
        for (int a = 0; a <= branchLayer; a++)
        {
            for (int i = offset[a]; i < offset[a + 1]; i++)
            {
                xyzKey[i] = i - offset[a];
            }
        }
        // initialize data layer
        sdfBuffer = new ComputeBuffer((int)(bufferSizeConstant * (1 << 3 * treeDepth)), 4);
        weightBuffer = new ComputeBuffer((int)(bufferSizeConstant * (1 << 3 * treeDepth)), 4);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, tailBufferID, tailBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, offsetBufferID, offsetBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, xyzKeyBufferID, xyzKeyBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, idChildArrBufferID, idChildArrBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, splitFlagBufferID, splitFlagBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, cameraMatrixBufferID, cameraMatrixBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, invCameraMatrixBufferID, invCameraMatrixBuffer);
        octreeShader.SetBuffer(SplitNodesFlagKernelID, depthBufferID, depthBuffer);

        octreeShader.SetBuffer(SplitNodesScanKernelID, splitFlagBufferID, splitFlagBuffer);
        octreeShader.SetBuffer(SplitNodesScanKernelID, tailBufferID, tailBuffer);

        octreeShader.SetBuffer(SplitNodesPropKernelID, tailBufferID, tailBuffer);
        octreeShader.SetBuffer(SplitNodesPropKernelID, splitFlagBufferID, splitFlagBuffer);
        octreeShader.SetBuffer(SplitNodesPropKernelID, offsetBufferID, offsetBuffer);
        octreeShader.SetBuffer(SplitNodesPropKernelID, xyzKeyBufferID, xyzKeyBuffer);
        octreeShader.SetBuffer(SplitNodesPropKernelID, idChildArrBufferID, idChildArrBuffer);

        octreeShader.SetBuffer(SplitNodesTailKernelID, tailBufferID, tailBuffer);
        octreeShader.SetBuffer(SplitNodesTailKernelID, splitFlagBufferID, splitFlagBuffer);

        octreeShader.SetBuffer(updateTopLayerKernelID, tailBufferID, tailBuffer);
        octreeShader.SetBuffer(updateTopLayerKernelID, offsetBufferID, offsetBuffer);
        octreeShader.SetBuffer(updateTopLayerKernelID, idChildArrBufferID, idChildArrBuffer);

        octreeShader.SetBuffer(updateSDFLayerKernelID, tailBufferID, tailBuffer);
        octreeShader.SetBuffer(updateSDFLayerKernelID, offsetBufferID, offsetBuffer);
        octreeShader.SetBuffer(updateSDFLayerKernelID, xyzKeyBufferID, xyzKeyBuffer);
        octreeShader.SetBuffer(updateSDFLayerKernelID, sdfBufferID, sdfBuffer);
        octreeShader.SetBuffer(updateSDFLayerKernelID, weightBufferID, weightBuffer);
        octreeShader.SetBuffer(updateSDFLayerKernelID, depthBufferID, depthBuffer);
        octreeShader.SetBuffer(updateSDFLayerKernelID, invCameraMatrixBufferID, invCameraMatrixBuffer);

        octreeShader.SetBuffer(surfacePredictKernelID, offsetBufferID, offsetBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, xyzKeyBufferID, xyzKeyBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, idChildArrBufferID, idChildArrBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, sdfBufferID, sdfBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, weightBufferID, weightBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, cameraMatrixBufferID, cameraMatrixBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, normalMapBufferID, normalMapBuffer);
        octreeShader.SetBuffer(surfacePredictKernelID, vertexMapBufferID, vertexMapBuffer);
        octreeShader.SetTexture(surfacePredictKernelID, outputBufferID, outputTexture);

        octreeShader.SetBuffer(DebugFunctionKernelID, resultBufferID, resultBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, depthBufferID, depthBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, invCameraMatrixBufferID, invCameraMatrixBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, xyzKeyBufferID, xyzKeyBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, offsetBufferID, offsetBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, sdfBufferID, sdfBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, weightBufferID, weightBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, idChildArrBufferID, idChildArrBuffer);
        octreeShader.SetBuffer(DebugFunctionKernelID, vertexMapBufferID, vertexMapBuffer);


        tailBuffer.SetData(tail);
        offsetBuffer.SetData(offset);
        xyzKeyBuffer.SetData(xyzKey);
        idChildArrBuffer.SetData(idChildArr);
        splitFlagBuffer.SetData(splitFlag);
        octreeShader.SetFloat(truncationDistID, truncationDist);
        octreeShader.SetFloat(maxSizeID, roomSize);
        octreeShader.SetMatrix(colorIntrinsicMatrixID, colorIntrinsicMatrix);
        octreeShader.SetInt(imageWidthID, imageWidth);
        octreeShader.SetInt(imageHeightID, imageHeight);
        octreeShader.SetInt(maxTSDFWeightID, maxTSDFWeight);
        octreeShader.SetInt(branchLayerID, branchLayer);
        octreeShader.SetInt(treeDepthID, treeDepth);
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
        tailBuffer.Release();
        offsetBuffer.Release();
        idChildArrBuffer.Release();
        xyzKeyBuffer.Release();
        splitFlagBuffer.Release();
        resultBuffer.Release();
        sdfBuffer.Release();
        weightBuffer.Release();
    }

    // Update is called once per frame
    void Update()
    {
        if (frame >= 300) return;
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
                defaultDepthArr[i] = testArr[i];
            }
            
            depthBuffer.SetData(defaultDepthArr);
            KinectFusion();
        }
        catch (System.Exception)
        {
            /*
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
            */
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
            /*
            int reductionBufferSize = Mathf.CeilToInt((float)imageWidth * imageHeight / waveGroupSize / 2.0f);
            float[] reductionBuffer = new float[reductionBufferSize * 32];
            ICPReductionBuffer.GetData(reductionBuffer);

            string outputTemp = "";
            for (int i = 0; i < reductionBufferSize; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    outputTemp += reductionBuffer[i * 32 + j] + " ";
                }
                outputTemp += "\n";
            }
            Debug.Log(outputTemp);
            */
        }

        computeShader.Dispatch(UpdateCameraMatrixID, 1, 1, 1);
        
        Matrix4x4[] cameraMatrixArr = new Matrix4x4[1];
        cameraMatrixBuffer.GetData(cameraMatrixArr);
        cameraMatrix = cameraMatrixArr[0];
        Debug.Log(cameraMatrix);


        //calculate TSDF
        //computeShader.Dispatch(TSDFUpdateID, voxelSize / 8, voxelSize / 8, voxelSize / 8);
        //render TSDF
        //computeShader.Dispatch(RenderTSDFID, imageWidth / 8, imageHeight / 8, 1);
        for (int i = branchLayer; i < treeDepth; i++)
        {
            octreeShader.SetInt(currentLayerID, i);
            octreeShader.Dispatch(SplitNodesFlagKernelID, (offset[i + 1] - offset[i] - 1) / 64 + 1, 1, 1);
            octreeShader.Dispatch(SplitNodesScanKernelID, 1, 1, 1);
            octreeShader.Dispatch(SplitNodesPropKernelID, (offset[i + 1] - offset[i] - 1) / 64 + 1, 1, 1);
            octreeShader.Dispatch(SplitNodesTailKernelID, 1, 1, 1);
        }

        int[] curTail = new int[tail.Length];
        tailBuffer.GetData(curTail);
        string output = "tail: ";
        for (int i = 0; i < curTail.Length; i++)
        {
            output += curTail[i] + " ";
        }
        Debug.Log(output);

        output = "offset: ";
        for (int i = 0; i < offset.Length; i++)
        {
            output += offset[i] + " ";
        }
        Debug.Log(output);

        for (int i = branchLayer - 1; i >= 0; i--)
        {
            octreeShader.SetInt(currentLayerID, i);
            octreeShader.Dispatch(updateTopLayerKernelID, (offset[i + 1] - offset[i] - 1) / 64 + 1, 1, 1);
        }

        Debug.Log((offset[treeDepth + 1] - offset[treeDepth] - 1) / 1024 + 1);
        octreeShader.SetInt(currentLayerID, treeDepth);
        octreeShader.Dispatch(updateSDFLayerKernelID, (offset[treeDepth + 1] - offset[treeDepth] - 1) / 1024 + 1, 1, 1);

        octreeShader.Dispatch(surfacePredictKernelID, imageWidth / 8, imageHeight / 8, 1);

        /*
        int[] idChildArr = new int[idChildArrBuffer.count];
        idChildArrBuffer.GetData(idChildArr);
        for (int i = 0; i < offset[3]; i++)
        {
            Debug.Log(i + " " + idChildArr[i]);
        }
        */

        /*
        octreeShader.SetInt(currentLayerID, branchLayer);
        octreeShader.Dispatch(DebugFunctionKernelID, 1, 1, 1);
        Vector4[] resultArr = new Vector4[1];
        resultBuffer.GetData(resultArr);
        Debug.Log(resultArr[0]);
        */
        
        /*
        float maxSize = roomSize;
        // split nodes and update child layers
        for (int i = branchLayer; i < treeDepth; i++)
        {
            if (tail[i] == 0) continue;
            int[] splitFlag = new int[tail[i]];
            for (int j = 0; j < tail[i]; j++)
            {
                float[] centerPos = findCenter(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, 0, i, xyzKey[offset[i] + j]);
                float sdf = calculateSDF(centerPos, testArr, cameraMatrix, colorIntrinsicMatrix);
                if (idChildArr[offset[i] + j] == -1 && Mathf.Abs(sdf) <= truncationDist + Mathf.Sqrt(3) * (maxSize / (float)(1 << (i + 1))))
                    splitFlag[j] = 1;
            }
            int[] shift = new int[splitFlag.Length];
            shift[0] = splitFlag[0];
            // TODO: replace with parallel prefix sum
            for (int j = 1; j < splitFlag.Length; j++)
            {
                shift[j] = shift[j - 1] + splitFlag[j];
            }

            for (int j = 0; j < tail[i]; j++)
            {
                if (splitFlag[j] == 1)
                {
                    idChildArr[offset[i] + j] = tail[i + 1] + ((shift[j] - 1) * 8);
                    int curKey = xyzKey[offset[i] + j] << 3;
                    for (int k = 0; k < 8; k++)
                    {
                        int pos = idChildArr[offset[i] + j] + k;
                        xyzKey[offset[i + 1] + pos] = curKey | k;
                    }
                }
            }
            tail[i + 1] += shift[shift.Length - 1] * 8;
        }

        // update top layers
        for (int i = branchLayer - 1; i >= 0; i--)
        {
            for (int j = 0; j < tail[i]; j++)
            {
                int childNode = j << 3;
                bool flag = true;
                for (int a = 0; a < 8; a++)
                {
                    if (idChildArr[offset[i + 1] + childNode + a] != -1)
                    {
                        flag = false;
                    }
                }
                if (flag)
                {
                    idChildArr[offset[i] + j] = -1;
                }
                else
                {
                    idChildArr[offset[i] + j] = childNode;
                }
            }
        }
        string output = "";
        for (int i = 0; i < tail.Length; i++)
        {
            output += tail[i] + " ";
        }
        Debug.Log(output);
        int count = 0;
        double tot = 0;
        float maxNum = 0;
        int countNum = 0;
        // update sdf for data layer
        for (int i = 0; i < tail[treeDepth]; i++)
        {
            float[] centerPos = findCenter(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, 0, treeDepth, xyzKey[offset[treeDepth] + i]);
            float sdf = calculateSDF(centerPos, testArr, cameraMatrix, colorIntrinsicMatrix);
            maxNum = Mathf.Max(maxNum, sdf);
            if (sdf > 0)
            {
                countNum++;
                tot += sdf;
            }
            
            float tsdf = Mathf.Clamp(sdf / truncationDist, -1, 1);
            dataLayerSDF[i] = (dataLayerSDF[i] * dataLayerWeights[i] + tsdf) / (dataLayerWeights[i] + 1);
            dataLayerWeights[i] = Mathf.Min(maxTSDFWeight, dataLayerWeights[i] + 1);
        }
        Debug.Log(count + " " + tail[treeDepth]);
        // Surface prediction
        Texture2D outputImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        float[] cameraPos = new float[] { cameraMatrix[0, 3], cameraMatrix[1, 3], cameraMatrix[2, 3] };
        Debug.Log(cameraMatrix[0, 3] + " " + cameraMatrix[1, 3] + " " + cameraMatrix[2, 3]);
        float epsilon = 1e-4f;
        float intersectEpsilon = 1e-6f;
        int totIterations = 0;
        int tempCount = 0;
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                vertexArr[i * imageWidth + j] = new Vector3();
                normalArr[i * imageWidth + j] = new Vector3();
                outputImage.SetPixel(j, imageHeight - i - 1, new Color());
                float[] rayPos = new float[] { cameraPos[0], cameraPos[1], cameraPos[2] };
                Vector4 tempRayDir = new Vector4((j - imageWidth / 2) / (float)(imageWidth / 2), (i - imageHeight / 2) / (float)(imageHeight / 2), 1, 0);
                tempRayDir = cameraMatrix * tempRayDir;
                float[] rayDir = new float[] { tempRayDir[0], tempRayDir[1], tempRayDir[2] };
                float length = (float)Mathf.Sqrt(rayDir[0] * rayDir[0] + rayDir[1] * rayDir[1] + rayDir[2] * rayDir[2]);
                rayDir[0] /= length;
                rayDir[1] /= length;
                rayDir[2] /= length;
                if (rayPos[0] < 0 || rayPos[0] > maxSize || rayPos[1] < 0 || rayPos[1] > maxSize || rayPos[2] < 0 || rayPos[2] > maxSize)
                {
                    rayPos = intersectCube(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, new float[] { rayPos[0], rayPos[1], rayPos[2] }, rayDir, new float[] { maxSize / 2.0f, maxSize / 2.0f, maxSize / 2.0f }, maxSize / 2.0f);
                    if (rayPos == null)
                        continue;
                }

                // get the node that currently contains rayPos
                int[] nodePrevData = findNode(new float[] { rayPos[0] + epsilon * rayDir[0], rayPos[1] + epsilon * rayDir[1], rayPos[2] + epsilon * rayDir[2] }, new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, treeDepth, branchLayer, idChildArr, offset);
                int nodeXYZKey = xyzKey[offset[nodePrevData[1]] + nodePrevData[0]];
                float[] nodePrevCenter = findCenter(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, 0, nodePrevData[1], nodeXYZKey);
                while (rayPos[0] >= -epsilon && rayPos[0] <= maxSize + epsilon && rayPos[1] >= -epsilon && rayPos[1] <= maxSize + epsilon && rayPos[2] >= -epsilon && rayPos[2] <= maxSize + epsilon)
                {
                    totIterations++;
                    float[] nextRayPos = intersectCube(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, new float[] { rayPos[0] + intersectEpsilon * rayDir[0], rayPos[1] + intersectEpsilon * rayDir[1], rayPos[2] + intersectEpsilon * rayDir[2] }, rayDir, nodePrevCenter, (maxSize / (float)(1 << (nodePrevData[1] + 1))));
                    if (nextRayPos == null)
                        break;
                    int[] nodeNextData = findNode(new float[] { nextRayPos[0] + epsilon * rayDir[0], nextRayPos[1] + epsilon * rayDir[1], nextRayPos[2] + epsilon * rayDir[2] }, new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, treeDepth, branchLayer, idChildArr, offset);
                    if (nodePrevData[1] == treeDepth && nodeNextData[1] == treeDepth)
                    {
                        if (dataLayerSDF[nodePrevData[0]] * dataLayerSDF[nodeNextData[0]] < 0)
                        {
                            float[] normalPrev = calculateNormal(rayPos, rayDir, new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, (maxSize / (float)(1 << nodePrevData[1])), epsilon, treeDepth, branchLayer, idChildArr, dataLayerSDF, nodePrevData, offset);
                            if (normalPrev == null)
                                break;
                            float[] normalNext = calculateNormal(nextRayPos, rayDir, new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, (maxSize / (float)(1 << nodePrevData[1])), epsilon, treeDepth, branchLayer, idChildArr, dataLayerSDF, nodeNextData, offset);
                            if (normalNext == null)
                                break;
                            float[] normal = new float[3];
                            for (int a = 0; a < 3; a++)
                            {
                                normal[a] = normalPrev[a] - dataLayerSDF[nodePrevData[0]] / (dataLayerSDF[nodePrevData[0]] - dataLayerSDF[nodeNextData[0]]) * (normalPrev[a] - normalNext[a]);
                            }
                            float norm = (float)Mathf.Sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
                            normal[0] /= norm;
                            normal[1] /= norm;
                            normal[2] /= norm;

                            int tempNodeNextXYZKey = xyzKey[offset[nodeNextData[1]] + nodeNextData[0]];
                            float[] tempNodeNextCenter = findCenter(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, 0, nodeNextData[1], tempNodeNextXYZKey);
                            float[] surfacePoint = new float[3];
                            for (int a = 0; a < 3; a++)
                            {
                                surfacePoint[a] = nodePrevCenter[a] - dataLayerSDF[nodePrevData[0]] / (dataLayerSDF[nodePrevData[0]] - dataLayerSDF[nodeNextData[0]]) * (nodePrevCenter[a] - tempNodeNextCenter[a]);
                            }
                            vertexArr[i * imageWidth + j] = new Vector3(surfacePoint[0], surfacePoint[1], surfacePoint[2]);
                            normalArr[i * imageWidth + j] = new Vector3(normal[0], normal[1], normal[2]);
                            tempCount++;

                            // normal[0] = Math.abs(normal[0]);
                            // normal[1] = Math.abs(normal[1]);
                            // normal[2] = Math.abs(normal[2]);
                            // int color = 0xFF000000 | (int)(normal[0] * 255) << 16 | (int)(normal[1] * 255) << 8 | (int)(normal[2] * 255);
                            float[] lightVector = new float[] { 0, 0, 1 };
                            //int intensity = (int)(Mathf.Abs(lightVector[0] * normal[0] + lightVector[1] * normal[1] + lightVector[2] * normal[2]) * 255);
                            //int color = intensity << 24 | intensity << 16 | intensity << 8 | 0xFF;
                            float intensity = Mathf.Abs(lightVector[0] * normal[0] + lightVector[1] * normal[1] + lightVector[2] * normal[2]);
                            outputImage.SetPixel(j, imageHeight - i - 1, new Color(intensity, intensity, intensity));

                            // outputImage.setRGB(j, i, 0xFFFFFFFF);
                            break;
                        }
                    }
                    int nodeNextXYZKey = xyzKey[offset[nodeNextData[1]] + nodeNextData[0]];
                    float[] nodeNextCenter = findCenter(new float[] { 0, 0, 0 }, new float[] { maxSize, maxSize, maxSize }, 0, nodeNextData[1], nodeNextXYZKey);
                    rayPos = nextRayPos;
                    nodePrevData = nodeNextData;
                    nodePrevCenter = nodeNextCenter;
                }
            }
        }
        Debug.Log(totIterations / (float)(imageHeight * imageWidth));
        outputImage.Apply();
        Graphics.Blit(outputImage, outputTexture);
        normalMapBuffer.SetData(normalArr);
        vertexMapBuffer.SetData(vertexArr);
        */
        rendererComponent.material.mainTexture = outputTexture;
        isTracking = true;
    }
    public static float calculateSDF(float[] pos, ushort[] depthBuffer, Matrix4x4 cameraMatrix, Matrix4x4 colorIntrinsicMatrix)
    {
        Vector4 pointPos = new Vector4(pos[0], pos[1], pos[2], 1);
        pointPos = cameraMatrix.inverse * pointPos;
        // transform point to camera's coordinate frame
        float[] posCameraFrame = new float[] { pointPos[0], pointPos[1], pointPos[2] };
        float projX = posCameraFrame[0] / posCameraFrame[2];
        float projY = posCameraFrame[1] / posCameraFrame[2];
        Vector4 imagePos = colorIntrinsicMatrix * new Vector4(projX, projY, 1, 1);
        int imageX = Mathf.RoundToInt(imagePos[0]);
        int imageY = Mathf.RoundToInt(imagePos[1]);
        if (imageX < 0 || imageX >= 640 || imageY < 0 || imageY >= 480 || pointPos[2] < 0) return -1e20f;
        if (depthBuffer[imageY * 640 + imageX] == 0) return -1e20f;
        float depth = depthBuffer[imageY * 640 + imageX] / 5000.0f;
        return depth * (float)Mathf.Sqrt(projX * projX + projY * projY + 1)
            - (float)Mathf.Sqrt(posCameraFrame[0] * posCameraFrame[0] + posCameraFrame[1] * posCameraFrame[1] + posCameraFrame[2] * posCameraFrame[2]);
    }
    public static float[] calculateNormal(float[] rayPos, float[] rayDir, float[] minCorner, float[] maxCorner, float nodeSize, float epsilon, int treeDepth, int branchLayer, int[] idChildArr, float[] dataLayerSDF, int[] nodePrevData, int[] offset)
    {
        int[] nodePosXData = findNode(new float[] { rayPos[0] + nodeSize + epsilon * rayDir[0], rayPos[1] + epsilon * rayDir[1], rayPos[2] + epsilon * rayDir[2] }, new float[] { minCorner[0], minCorner[1], minCorner[2] }, new float[] { maxCorner[0], maxCorner[1], maxCorner[2] }, treeDepth, branchLayer, idChildArr, offset);
        int[] nodePosYData = findNode(new float[] { rayPos[0] + epsilon * rayDir[0], rayPos[1] + nodeSize + epsilon * rayDir[1], rayPos[2] + epsilon * rayDir[2] }, new float[] { minCorner[0], minCorner[1], minCorner[2] }, new float[] { maxCorner[0], maxCorner[1], maxCorner[2] }, treeDepth, branchLayer, idChildArr, offset);
        int[] nodePosZData = findNode(new float[] { rayPos[0] + epsilon * rayDir[0], rayPos[1] + epsilon * rayDir[1], rayPos[2] + nodeSize + epsilon * rayDir[2] }, new float[] { minCorner[0], minCorner[1], minCorner[2] }, new float[] { maxCorner[0], maxCorner[1], maxCorner[2] }, treeDepth, branchLayer, idChildArr, offset);
        int[] nodeNegXData = findNode(new float[] { rayPos[0] - nodeSize + epsilon * rayDir[0], rayPos[1] + epsilon * rayDir[1], rayPos[2] + epsilon * rayDir[2] }, new float[] { minCorner[0], minCorner[1], minCorner[2] }, new float[] { maxCorner[0], maxCorner[1], maxCorner[2] }, treeDepth, branchLayer, idChildArr, offset);
        int[] nodeNegYData = findNode(new float[] { rayPos[0] + epsilon * rayDir[0], rayPos[1] - nodeSize + epsilon * rayDir[1], rayPos[2] + epsilon * rayDir[2] }, new float[] { minCorner[0], minCorner[1], minCorner[2] }, new float[] { maxCorner[0], maxCorner[1], maxCorner[2] }, treeDepth, branchLayer, idChildArr, offset);
        int[] nodeNegZData = findNode(new float[] { rayPos[0] + epsilon * rayDir[0], rayPos[1] + epsilon * rayDir[1], rayPos[2] - nodeSize + epsilon * rayDir[2] }, new float[] { minCorner[0], minCorner[1], minCorner[2] }, new float[] { maxCorner[0], maxCorner[1], maxCorner[2] }, treeDepth, branchLayer, idChildArr, offset);
        float normalX = 0;
        float normalY = 0;
        float normalZ = 0;
        if ((nodePosXData[1] != treeDepth && nodeNegXData[1] != treeDepth) || (nodePosYData[1] != treeDepth && nodeNegYData[1] != treeDepth) || (nodePosZData[1] != treeDepth && nodeNegZData[1] != treeDepth))
        {
            return null;
        }
        if (nodePosXData[1] == treeDepth)
        {
            normalX = dataLayerSDF[nodePosXData[0]] - dataLayerSDF[nodePrevData[0]];
        }
        else
        {
            normalX = dataLayerSDF[nodePrevData[0]] - dataLayerSDF[nodeNegXData[0]];
        }
        if (nodePosYData[1] == treeDepth)
        {
            normalY = dataLayerSDF[nodePosYData[0]] - dataLayerSDF[nodePrevData[0]];
        }
        else
        {
            normalY = dataLayerSDF[nodePrevData[0]] - dataLayerSDF[nodeNegYData[0]];
        }
        if (nodePosZData[1] == treeDepth)
        {
            normalZ = dataLayerSDF[nodePosZData[0]] - dataLayerSDF[nodePrevData[0]];
        }
        else
        {
            normalZ = dataLayerSDF[nodePrevData[0]] - dataLayerSDF[nodeNegZData[0]];
        }
        float[] normal = new float[] { normalX, normalY, normalZ };
        float len = (float)Mathf.Sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        normal[0] /= len;
        normal[1] /= len;
        normal[2] /= len;
        return normal;
    }
    public static float[] intersectCube(float[] minCorner, float[] maxCorner, float[] rayPos, float[] rayDir, float[] nodeCenter, float nodeSize)
    {
        float minTime = 1e20f;
        bool hasIntersect = false;
        for (int i = 0; i < 3; i++)
        {
            for (int j = -1; j <= 1; j += 2)
            {
                float[] planeCenter = new float[] { nodeCenter[0], nodeCenter[1], nodeCenter[2] };
                planeCenter[i] += j * nodeSize;
                float[] planeNormal = new float[3];
                planeNormal[i] = j;
                float t = intersectPlane(planeCenter, planeNormal, rayPos, rayDir);
                float[] intersectPos = new float[] { rayPos[0] + t * rayDir[0], rayPos[1] + t * rayDir[1], rayPos[2] + t * rayDir[2] };
                bool flag = true;
                for (int a = 0; a < 3; a++)
                {
                    if (a == i) continue;
                    if (Mathf.Abs(planeCenter[a] - intersectPos[a]) >= nodeSize + 1e-3)
                        flag = false;
                }
                if (t > 0 && minTime > t && flag)
                {
                    hasIntersect = true;
                    minTime = t;
                }
            }
        }
        if (hasIntersect)
        {
            float[] intersectPos = new float[] { rayPos[0] + minTime * rayDir[0], rayPos[1] + minTime * rayDir[1], rayPos[2] + minTime * rayDir[2] };
            return intersectPos;
        }
        return null;
    }
    public static float intersectPlane(float[] planeCenter, float[] planeNormal, float[] rayPos, float[] rayDir)
    {
        float denom = planeNormal[0] * rayDir[0] + planeNormal[1] * rayDir[1] + planeNormal[2] * rayDir[2];
        if (Mathf.Abs(denom) > 1e-3f)
        {
            float t = ((planeCenter[0] - rayPos[0]) * planeNormal[0] + (planeCenter[1] - rayPos[1]) * planeNormal[1] + (planeCenter[2] - rayPos[2]) * planeNormal[2]) / denom;
            return t;
        }
        return -1;
    }
    public static int calculateXYZKey(float[] minCorner, float[] maxCorner, float[] pos, int maxDepth)
    {
        int xyzKey = 0;
        for (int i = 0; i < maxDepth; i++)
        {
            float[] centerPos = new float[] { (minCorner[0] + maxCorner[0]) / 2, (minCorner[1] + maxCorner[1]) / 2, (minCorner[2] + maxCorner[2]) / 2 };
            int curKey = 0;
            if (pos[0] < centerPos[0])
            {
                maxCorner[0] = centerPos[0];
            }
            else
            {
                minCorner[0] = centerPos[0];
                curKey |= 0b001;
            }
            if (pos[1] < centerPos[1])
            {
                maxCorner[1] = centerPos[1];
            }
            else
            {
                minCorner[1] = centerPos[1];
                curKey |= 0b010;
            }
            if (pos[2] < centerPos[2])
            {
                maxCorner[2] = centerPos[2];
            }
            else
            {
                minCorner[2] = centerPos[2];
                curKey |= 0b100;
            }
            xyzKey = (xyzKey << 3) | curKey;
        }
        return xyzKey;
    }
    public static int[] findNode(float[] pos, float[] minCorner, float[] maxCorner, int maxDepth, int branchLayer, int[] idChildArr, int[] offset)
    {
        // calculate shuffled xyz key of pos
        int xyzKey = calculateXYZKey(minCorner, maxCorner, pos, maxDepth);
        for (int i = 0; i < branchLayer; i++)
        {
            int curNodeKey = xyzKey >> (maxDepth - i) * 3;
            if (idChildArr[offset[i] + curNodeKey] == -1)
            {
                return new int[] { curNodeKey, i };
            }
        }

        int curDepth = branchLayer;
        int curNode = xyzKey >> (maxDepth - branchLayer) * 3;
        while (curDepth < maxDepth && idChildArr[offset[curDepth] + curNode] != -1)
        {
            curNode = idChildArr[offset[curDepth] + curNode] + ((xyzKey >> 3 * (maxDepth - curDepth - 1)) & 0b111);
            curDepth++;
        }
        return new int[] { curNode, curDepth };
    }
    public static float[] findCenter(float[] minCorner, float[] maxCorner, int curDepth, int maxDepth, int xyzKey)
    {
        if (curDepth == maxDepth) return new float[] { (minCorner[0] + maxCorner[0]) / 2, (minCorner[1] + maxCorner[1]) / 2, (minCorner[2] + maxCorner[2]) / 2 };
        if ((xyzKey & 1 << (maxDepth - curDepth - 1) * 3) != 0)
        {
            minCorner[0] = (maxCorner[0] + minCorner[0]) / 2.0f;
        }
        else
        {
            maxCorner[0] = (maxCorner[0] + minCorner[0]) / 2.0f;
        }

        if ((xyzKey & 1 << (maxDepth - curDepth - 1) * 3 + 1) != 0)
        {
            minCorner[1] = (maxCorner[1] + minCorner[1]) / 2.0f;
        }
        else
        {
            maxCorner[1] = (maxCorner[1] + minCorner[1]) / 2.0f;
        }

        if ((xyzKey & 1 << (maxDepth - curDepth - 1) * 3 + 2) != 0)
        {
            minCorner[2] = (maxCorner[2] + minCorner[2]) / 2.0f;
        }
        else
        {
            maxCorner[2] = (maxCorner[2] + minCorner[2]) / 2.0f;
        }
        return findCenter(minCorner, maxCorner, curDepth + 1, maxDepth, xyzKey);
    }
}
