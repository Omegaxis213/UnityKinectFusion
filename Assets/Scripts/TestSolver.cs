using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestSolver : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        float[,] ICPSharedData = new float[32, 6];
        /*
        for (int b = 0; b < 32; b++)
        {
            ICPSharedData[b,0] = 0;
        }
        for (int a = 0; a < reductionGroupSize; a++)
        {
            for (int b = 0; b < 32; b++)
            {
                ICPSharedData[b, 0] += ICPReductionBuffer[a * 32 + b];
            }
        }
        for (int a = 0; a < 6; a++)
        {
            for (int b = a; b < 6; b++)
            {
                ICPSharedData[a * 6 + b, 1] = ICPSharedData[a * 6 - a * (a - 1) / 2 + b - a, 0];
                ICPSharedData[b * 6 + a, 1] = ICPSharedData[a * 6 - a * (a - 1) / 2 + b - a, 0];
                ICPSharedData[a * 6 + b, 2] = 0;
                ICPSharedData[b * 6 + a, 2] = 0;
            }
        }
        

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = 0;
                if (j == i)
                {
                    for (int k = 0; k < j; k++)
                        sum += ICPSharedData[j * 6 + k, 2] * ICPSharedData[j * 6 + k, 2];
                    ICPSharedData[j * 6 + j, 2] = Mathf.Sqrt(ICPSharedData[j * 6 + j, 1] - sum);
                }
                else
                {
                    for (int k = 0; k < j; k++)
                        sum += ICPSharedData[i * 6 + k, 2] * ICPSharedData[j * 6 + k, 2];
                    ICPSharedData[i * 6 + j, 2] = (ICPSharedData[i * 6 + j, 1] - sum) / ICPSharedData[j * 6 + j, 2];
                }
            }
        }
        */

        ICPSharedData[0, 1] = 4;
        ICPSharedData[1, 1] = 12;
        ICPSharedData[2, 1] = -16;
        ICPSharedData[3, 1] = 12;
        ICPSharedData[4, 1] = 37;
        ICPSharedData[5, 1] = -43;
        ICPSharedData[6, 1] = -16;
        ICPSharedData[7, 1] = -43;
        ICPSharedData[8, 1] = 98;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                ICPSharedData[i * 3 + j, 2] = 0;
            }
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = 0;
                if (j == i)
                {
                    for (int k = 0; k < j; k++)
                        sum += ICPSharedData[j * 3 + k, 2] * ICPSharedData[j * 3 + k, 2];
                    ICPSharedData[j * 3 + j, 2] = Mathf.Sqrt(ICPSharedData[j * 3 + j, 1] - sum);
                }
                else
                {
                    for (int k = 0; k < j; k++)
                        sum += ICPSharedData[i * 3 + k, 2] * ICPSharedData[j * 3 + k, 2];
                    ICPSharedData[i * 3 + j, 2] = (ICPSharedData[i * 3 + j, 1] - sum) / ICPSharedData[j * 3 + j, 2];
                }
            }
        }
        string output = "";
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                output += ICPSharedData[i * 3 + j, 2] + " ";
            }
            output += "\n";
        }
        Debug.Log(output);
        ICPSharedData[21, 0] = 32;
        ICPSharedData[22, 0] = 43;
        ICPSharedData[23, 0] = 12;
        for (int i = 0; i < 3; i++)
        {
            float temp = ICPSharedData[21 + i, 0];
            for (int j = 0; j < i; j++)
            {
                temp -= ICPSharedData[i * 3 + j, 2] * ICPSharedData[j, 3];
            }
            temp /= ICPSharedData[i * 3 + i, 2];
            ICPSharedData[i, 3] = temp;
        }
        output = "";
        for (int i = 0; i < 3; i++)
            output += ICPSharedData[i, 3] + " ";
        Debug.Log(output);
        for (int i = 2; i >= 0; i--)
        {
            float temp = ICPSharedData[i, 3];
            for (int j = 2; j > i; j--)
            {
                temp -= ICPSharedData[j * 3 + i, 2] * ICPSharedData[j, 4];
            }
            temp /= ICPSharedData[i * 3 + i, 2];
            ICPSharedData[i, 4] = temp;
        }
        output = "";
        for (int i = 0; i < 3; i++)
            output += ICPSharedData[i, 4] + " ";
        Debug.Log(output);

        /*
        for (int i = 0; i < 6; i++)
        {
            float temp = ICPSharedData[21 + i, 0];
            for (int j = 0; j < i - 1; j++)
            {
                temp -= ICPSharedData[i * 6 + j, 2] * ICPSharedData[j, 3];
            }
            temp /= ICPSharedData[i * 6 + i, 2];
            ICPSharedData[i, 3] = temp;
        }

        for (int i = 5; i >= 0; i--)
        {
            float temp = ICPSharedData[i, 3];
            for (int j = 5; j > i; j--)
            {
                temp -= ICPSharedData[j * 6 + i, 2] * ICPSharedData[j, 4];
            }
            temp /= ICPSharedData[i * 6 + i, 2];
            ICPSharedData[i, 4] = temp;
        }
        float[] CholeskyBuffer = new float[6];
        for (int i = 0; i < 6; i++)
            CholeskyBuffer[i] = ICPSharedData[i, 4];
        string output = "";
        for (int i = 0; i < 6; i++)
            output += CholeskyBuffer[i] + " ";
        Debug.Log(output);
        */
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
