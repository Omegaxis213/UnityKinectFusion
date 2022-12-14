using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Emgu.CV;

public class NewBehaviourScript : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        int size = 10000;
        /*
        float[,] leftMat = new float[6, 6];
        float[] rightVal = new float[6];
        */
        
        float[,] leftMat = new float[size * 2, 6];
        float[] rightVal = new float[size * 2];
        
        for (int i = 0; i < size; i++)
        {
            //Vector3 prevNormal = new Vector3(1 + Random.Range(0f, .0001f), Random.Range(0f, .0001f), Random.Range(0f, .0001f)).normalized;
            //Vector3 prevNormal = new Vector3(1 + Random.Range(0f, .01f), Random.Range(0f, .01f), Random.Range(0f, .01f)).normalized;
            Vector3 prevNormal = new Vector3(1 + Random.Range(-1.5f, 1.5f), Random.Range(-1.5f, 1.5f), Random.Range(-1.5f, 1.5f)).normalized;
            //Vector3 prevNormal = new Vector3(1, 0, 0).normalized;
            Vector3 prevVertex = new Vector3(Random.Range(-100, 100), Random.Range(-100, 100), Random.Range(-100, 100));
            Vector3 estimateVertex = prevVertex + new Vector3(Random.Range(-3f, 3f), Random.Range(-3f, 3f), Random.Range(-3f, 3f));
            //Vector3 estimateVertex = prevVertex;
            float[] ATransposeMatrix = new float[] { estimateVertex.y * prevNormal.z - estimateVertex.z * prevNormal.y,
                                                             estimateVertex.z * prevNormal.x - estimateVertex.x * prevNormal.z,
                                                             estimateVertex.x * prevNormal.y - estimateVertex.y * prevNormal.x,
                                                             prevNormal.x,
                                                             prevNormal.y,
                                                             prevNormal.z};
            //float bScalar = prevNormal.x * (prevVertex.x - estimateVertex.x) + prevNormal.y * (prevVertex.y - estimateVertex.y) + prevNormal.z * (prevVertex.z - estimateVertex.z);
            float bScalar = prevNormal.x * (estimateVertex.x - prevVertex.x) + prevNormal.y * (estimateVertex.y - prevVertex.y) + prevNormal.z * (estimateVertex.z - prevVertex.z);
            for (int a = 0; a < 6; a++)
                leftMat[i, a] = ATransposeMatrix[a];
            rightVal[i] = bScalar;
            /*
            for (int a = 0; a < 6; a++)
            {
                rightVal[a] += bScalar * ATransposeMatrix[a];
                for (int b = 0; b < 6; b++)
                {
                    leftMat[a, b] += ATransposeMatrix[a] * ATransposeMatrix[b];
                }
            }
            */
        }

        unsafe
        {
            fixed (float* ptrTwo = rightVal)
            {
                fixed (float* ptr = leftMat)
                {
                    /*
                    string printOut = "";
                    for (int a = 0; a < 6 * size; a++)
                    {
                        for (int b = 0; b < 6; b++)
                        {
                            printOut += leftMat[a, b] + " ";
                        }
                        printOut += "\n";
                    }
                    Debug.Log(printOut);
                    printOut = "";
                    for (int a = 0; a < 6 * size; a++)
                    {
                        printOut += rightVal[a] + " ";
                    }
                    Debug.Log(printOut);
                    */
                    Mat leftArr = new Mat(size, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptr), 6 * 4);
                    Mat rightValArr = new Mat(size, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptrTwo), 4);
                    Mat result = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                    Debug.Log(CvInvoke.Solve(leftArr, rightValArr, result, Emgu.CV.CvEnum.DecompMethod.Svd) ? "Found a solution" : "No solution found");
                    float[] tempArr = new float[6];
                    result.CopyTo(tempArr);
                    Matrix4x4 incMat = new Matrix4x4(new Vector4(1, -tempArr[2], tempArr[1], 0),
                                                     new Vector4(tempArr[2], 1, -tempArr[0], 0),
                                                     new Vector4(-tempArr[1], tempArr[0], 1, 0),
                                                     new Vector4(tempArr[3], tempArr[4], tempArr[5], 1));
                    Debug.Log("incremental: " + incMat);
                }
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
