#include <iostream>
#include<emmintrin.h>
#include<time.h>
#include<Windows.h>
#include <immintrin.h>
#include <pthread.h>

using namespace std;
const int N = 1000;
float elm[N][N] = {0};
float ans[N][N] = {0};

const int thread_count = 5;
const float eps = 1e-6;

struct param_gauss{
    float **m;
    int n;
    int r;
};

void gaussian_naive(float **m, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = i + 1; j < n; j++)
            m[i][j] = m[i][j] / m[i][i];
        m[i][i] = 1;
        for(int j = i + 1; j < n; j++)
        {
            for(int k = i + 1; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];
            m[j][i] = 0;
         }
    }
    return;
}

// SSE SIMD optimization
void gaussian_sse(float **m, int n)
{
    __m128 t1, t2, t3;
    for(int i = 0; i < n; i++)
    {
        float t[4] = {m[i][i], m[i][i], m[i][i], m[i][i]};
        t2 = _mm_loadu_ps(t);
        int j = i + 1;
        for(j; j < n - 4; j += 4)
        {
            t1 = _mm_loadu_ps(m[i] + j);
            t3 = _mm_div_ps(t1, t2);
            _mm_storeu_ps(m[i] + j, t3);
        }
        for(j; j < n; j++)
            m[i][j] = m[i][j] / m[i][i];
        m[i][i] = 1;
        for(int j = i + 1; j < n; j++)
        {
            float temp2[4] = {m[j][i], m[j][i], m[j][i], m[j][i]};
            t2 = _mm_loadu_ps(temp2);
            int k = i + 1;
            for(k; k < n - 4; k+=4)
            {
                t1 = _mm_loadu_ps(m[i] + k);
                t1 = _mm_mul_ps(t1, t2);
                t3 = _mm_loadu_ps(m[j] + k);
                t3 = _mm_sub_ps(t3, t1);
                _mm_storeu_ps(m[j] + k, t3);
            }
            for(k; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];
            m[j][i] = 0;
         }
    }
    return;
}

pthread_barrier_t barrier;

void* gaussian_sse_parallel_1(void* param)
{
    struct param_gauss *p = (struct param_gauss *) param;
    float ** m = p->m;
    int n = p->n;
    int r = p->r;
    __m128 t1, t2, t3;
    for(int i = 0; i < n; i++)
    {
        if(r == 0)
        {
            float t[4] = {m[i][i], m[i][i], m[i][i], m[i][i]};
            t2 = _mm_loadu_ps(t);
            int j = i + 1;
            for(j; j < n - 4; j += 4)
            {
                t1 = _mm_loadu_ps(m[i] + j);
                t3 = _mm_div_ps(t1, t2);
                _mm_storeu_ps(m[i] + j, t3);
            }
            for(j; j < n; j++)
                m[i][j] = m[i][j] / m[i][i];
            m[i][i] = 1;
        }

        // wait until the thread 0 finish its job
        pthread_barrier_wait(&barrier);

        int gap = (n - 1 - i) / (thread_count);
        int start = r * gap + i + 1;
        int last = -1;
        if(r == (thread_count - 1))
            last = n;
        else
            last = start + gap;
//        cout<<start<<" "<<last<<endl;

        for(int j = start; j < last; j++)
        {
            float temp2[4] = {m[j][i], m[j][i], m[j][i], m[j][i]};
            t2 = _mm_loadu_ps(temp2);
            int k = i + 1;
            for(k; k < n - 4; k+=4)
            {
                t1 = _mm_loadu_ps(m[i] + k);
                t1 = _mm_mul_ps(t1, t2);
                t3 = _mm_loadu_ps(m[j] + k);
                t3 = _mm_sub_ps(t3, t1);
                _mm_storeu_ps(m[j] + k, t3);
            }
            for(k; k < n; k++)
                m[j][k] = m[j][k] - m[i][k] * m[j][i];

            m[j][i] = 0;
         }
         pthread_barrier_wait(&barrier);
    }
    pthread_exit(nullptr);
}

void reset(float **test)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            test[i][j] = elm[i][j];
        }
    }
    return;
}

bool test_ans(float ** test)
{
    for(int i =0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            if((-eps > test[i][j] - ans[i][j]) || (eps < test[i][j] - ans[i][j]))
            {
                cout<<i<<" "<<j<<" "<<test[i][j]<<" "<<ans[i][j]<<endl;
//                return false;
            }
        }
    return true;
}

int main()
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            elm[i][j] = (rand() % 50);
        }
    }
    float** test = new float*[N];
    for (int i = 0; i < N; i++)
    {
        test[i] = new float[N];
    }

    reset(test);

    srand(time(NULL));
    LARGE_INTEGER timeStart;
    LARGE_INTEGER timeEnd;

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    double quadpart = (double)frequency.QuadPart;

    //naive
    QueryPerformanceCounter(&timeStart);
    gaussian_naive(test, N);
    QueryPerformanceCounter(&timeEnd);
    double _Simple = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
    printf("Simple:%f\n", _Simple);
    for(int i =0; i < N; i++)
        for(int j = 0; j < N; j++)
            ans[i][j] = test[i][j];
    cout << endl;
//    for(int i =0; i < 10; i++)
//    {
//        for(int j = 0; j < 10; j++)
//            cout << test[i][j] << " ";
//        cout <<endl;
//    }
//    cout << endl;
    reset(test);


    //SSE
    QueryPerformanceCounter(&timeStart);
    gaussian_sse(test, N);
    QueryPerformanceCounter(&timeEnd);
    double _SSE_Gauss = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
     bool correct = test_ans(test);
    if(correct)
        cout<<"good"<<endl;
    printf("SSE_Gauss:%f\n", _SSE_Gauss);
    cout << endl;
    reset(test);

    //Parallel
    pthread_barrier_init(&barrier,NULL,thread_count);
    pthread_t thread[thread_count];
    struct param_gauss thread_param[thread_count];
    for(int i=0;i<thread_count;i++){
        thread_param[i].m = test;
        thread_param[i].r = i;
        thread_param[i].n = N;
    }
    QueryPerformanceCounter(&timeStart);
    for(int i=0;i<thread_count;i++){
        pthread_create(&thread[i],nullptr,gaussian_sse_parallel_1,(void*)(&thread_param[i]));
    }
    for(int i=0;i<thread_count;i++){
        pthread_join(thread[i],nullptr);
    }
    correct = test_ans(test);
    if(correct)
        cout<<"good"<<endl;
//    for(int i =0; i < 10; i++)
//    {
//        for(int j = 0; j < 10; j++)
//            cout << test[i][j] << " ";
//        cout <<endl;
//    }
    QueryPerformanceCounter(&timeEnd);
    double _Par_Gauss = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
    printf("Par_Gauss:%f\n", _Par_Gauss);
    cout << endl;
    reset(test);


    system("pause");
    return 0;
}

