#include <opencv2/opencv.hpp>
#include "mid_level.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <queue>
#include <memory>
using namespace std;
using namespace cv;

//-------------------------------------------------//


// 'feature' is a class, contain image info. and image features
ofstream file("debug.txt");

class clusterNode
{
public:
    feature* feat;
    clusterNode* next;
    clusterNode() { feat = NULL; next = NULL; }
    clusterNode( feature* f ) { feat = f; next = NULL; }
};

class cluster
{
public:
    int count;
    clusterNode* head;
    cluster() { count = 0; head = NULL; }
    cluster( feature* f )
    {
        count = 1;
        head = new clusterNode( f );
        head->next = NULL;
    }

    void insert( clusterNode* node )
    {
        count++;
        node->next = head;
        head = node;
    }
};

class affinityNode
{
public:
    cluster* a;
    cluster* b;
    double affi;
    affinityNode( cluster* aa , cluster* bb, double af )
    {
        a = aa;
        b = bb;
        affi = af;
    }

    friend bool operator < ( const affinityNode& a, const affinityNode& b )
    {
        return a.affi < b.affi;
    }
};

bool cmp( sortAssistant a, sortAssistant b )
{
    return a.dist < b.dist;
}

//-------------------------------------------------//

double getEuclidDistSquare( feature& a, feature& b )
{
    // calculate squared euclidean-distance b/w feat.a and feat.b
    double *ptr1, *ptr2, sum=0;
    ptr1=a.feat.ptr<double>(0);
    ptr2=b.feat.ptr<double>(0);
    int l=a.feat.cols;
//#pragma omp parallel for reduction(+:sum)
    for (int i=0;i<l;i++)
    {
        sum=sum+(ptr1[i]-ptr2[i])*(ptr1[i]-ptr2[i]);
    }
    // cout << f << " ";
    return sum;
}

void clustering( vector<feature>& feats, vector< vector<sortAssistant> >& KNN_array, vector< vector<feature> >& cluster_result )
{
    // Zhang W, Wang X, Zhao D, et al.
    // Graph degree linkage: Agglomerative clustering on a directed graph

    /*
     *
     * input: 'vector<feature>& feats' - all features distracted from patches
     * output: 'vector< vector<feature> >& cluster_result' - clustered features,
     *         'vector< vector<sortAssistant> >& KNN_array' - for further use
     *
     */
    int numOfFeats;

    Mat allWeight;
    vector<cluster*> allCluster;
    vector<cluster*> result;
    vector<feature> mid_level_feats;


    midLevelSelect( feats, mid_level_feats, 5 );
    /* select those feats which are mid-level
     * mid-level: the sum of KNN-dist local in the 5 level ( out of 10 )
     *
     * do cluster base on the mid-level features (patches)
     */

    numOfFeats = mid_level_feats.size();
    allWeight.create( numOfFeats, numOfFeats, CV_64FC1 );

    for( int i = 0; i < numOfFeats; i++ )
        mid_level_feats[i].index = i;

    double* weight;
    double sigmaSquare = 0.0;
    weight = (double*)allWeight.data;

    calcAllPairDist( weight, mid_level_feats, &sigmaSquare, 30, KNN_array );
    /* ( not the final weights )
     * w_ij = dist(i,j)^2, if j in KNN of i;
     *      = 0, otherwise.
     */

    /* debug code
     * ofstream KNN_file("KNN.txt");
    for( int i = 0; i < numOfFeats; i++ )
    {
        for( int j = 0; j < KNN_array[i].size() ; j++ )
            KNN_file << KNN_array[i][j].index << " " << KNN_array[i][j].im_index << " " << KNN_array[i][j].pos_index << " " << KNN_array[i][j].dist << " ";
        KNN_file << endl;
    } */

    calcWeight( weight, numOfFeats, 30, &sigmaSquare );
    /* w_ij is final weight of edge(i,j)
     * w_ij = exp( - dist(i,j)^2 / sigma^2 ), if j in KNN of i;
     *      = 0, otherwise.
     */

    // print( allWeight );

    hierachy( mid_level_feats, allWeight, allCluster, result );
    // do hierachy-cluster, rough -> fine

    for( int i = 0; i < result.size(); i++ )
    {
        // output
        clusterNode* ptr = result[i]->head;
        vector<feature> clst;
        while( ptr != NULL )
        {
            clst.push_back( *(ptr->feat) );
            ptr = ptr->next;
        }
        cluster_result.push_back( clst );
    }

    file << "total: " << result.size() << endl;

    for( int i = 0; i < result.size(); i++ )
        file << result[i]->count << endl;    /* w_ij = exp( - dist(i,j)^2 / sigma^2 ), if j in KNN of i;
     *      = 0, otherwise.
     */

    /*initialCluster( mid_level_feats, allCluster );



    doCluster( allCluster, 4, allWeight );

    cout << 5;*/
    // print( allCluster );
    // cout << 5;

}

void hierachy( vector<feature>& allFeats, Mat& allWeight, vector<cluster*>& allCluster, vector<cluster*>& result )
{
    vector<cluster*> buffer[2];
    int a, b;
    cluster* clst;
    initialCluster( allFeats, allCluster );
    // aggregate similar features as initial cluster

    doCluster( allCluster, 4, allWeight );
    for( int i = 0; i < allCluster.size(); i++ )
        buffer[0].push_back(allCluster[i]);

    for( int i = 0; i < 10; i++ )
    {
        // 10-level of cluster
        a = i % 2;
        b = a == 0 ? 1 : 0;

        while( !buffer[a].empty() )
        {
            clst = buffer[a].back();
            buffer[a].pop_back();
            if( clst->count < 15 )
                continue; // abandon cluster with few features
            else if( clst->count < 30 )
                result.push_back( clst ); // do not split cluster with 15~29 features
            else
            {
                // split the cluster in to 4 smaller cluster
                vector<cluster*> split;
                clusterNode* ptr = clst->head;
                while( ptr != NULL )
                {
                    split.push_back( new cluster( ptr->feat ) );
                    ptr = ptr->next;
                }
                doCluster( split, 4, allWeight );
                for( int i = 0; i < split.size(); i++ )
                    buffer[b].push_back( split[i] );
            }
        }
    }
    for( int i = 0; i < buffer[0].size(); i++ )
        if( buffer[0][i]->count >= 15 )
            result.push_back( buffer[0][i] );
    for( int i = 0; i < buffer[1].size(); i++ )
        if( buffer[1][i]->count >= 15 )
            result.push_back( buffer[1][i] );

}



void midLevelSelect( vector<feature>& all_feats, vector<feature>& mid_level_feats, int level )
{
    /* mid-level: the sum of KNN-dist local in the 5 level ( out of 10 )
     */
    int numOfFeats = all_feats.size();
    Mat allDist( numOfFeats, numOfFeats, CV_64FC1 );
    double* dist = (double*)allDist.data;
    double sigmaSquare = 0.0; // useless

    calcAllPairDist( dist, all_feats, &sigmaSquare, (int)(0.3*numOfFeats));
    // calculate dist/KNN

    Mat Ones( numOfFeats, 1, CV_64FC1,Scalar(1.0) );
    Mat reslt = allDist * Ones;
    // calculate the sum of KNN-dist

    double* tmp = (double*)(reslt).data;
    double max = -1.0;
    double min = 99999.9;
    for( int i = 0; i < numOfFeats; i++ )
    {
        if( tmp[i] > max )
            max = tmp[i];
        else if( tmp[i] < min )
            min = tmp[i];
    }
    for (int i=0;i<numOfFeats;i++)
    {
        all_feats[i].sAUC=(tmp[i]-min)*10.0/(max-min);
    }
    // file << max << " " << min << endl;

    // selecet mid-level features
    double step = ( max - min ) / 10;
    double lower = min + level*step, upper = lower + step+step;//+step*2
    int cnt = 0;
    for( int j = 0; j < numOfFeats; j++ )
        if( tmp[j] >= lower && tmp[j] <= upper )
        {
            // file << j << endl;
            mid_level_feats.push_back( all_feats[j] );
            cnt++;
        }
    file << "-cnt = " << cnt << "----" << endl;



}



void doCluster( vector<cluster*>& allCluster, int k, Mat& w )
{
    // combine all cluster until ( # of cluster == k )
    priority_queue<affinityNode> Q;
    double affi;
    cluster* a;
    cluster* b;

    /* initial priority-queue Q with all possible pairs of all clusters
     * key of Q is affinity b/w two cluster
     */
    for( int i = 0; i < allCluster.size(); i++ )
    {
        for( int j = i+1; j < allCluster.size(); j++ )
        {
            a = allCluster[i];
            b = allCluster[j];
            affi = affinity( a, b, w );
            affinityNode n( a, b, affi );
            Q.push( n );
        }
    }

    while( allCluster.size() > k )
    {
        /* distract pair of cluster with highest affinity and merge them
         * until number of clusters is fewer than k
         */
        affinityNode n = Q.top();
        Q.pop();
        bool founda = find( allCluster.begin(), allCluster.end(), n.a) != allCluster.end();
        bool foundb = find( allCluster.begin(), allCluster.end(), n.b) != allCluster.end();
        bool notTooLarge = true; // ( n.a->count < numOfFeats/5 ) && ( n.b->count < numOfFeats/5 );
        // cout << founda << foundb << endl;
        if( founda && foundb && notTooLarge )
        {
            // cout << n.a->head->feat->index << n.b->head->feat->index << endl;
            for( vector<cluster*>::iterator it = allCluster.begin(); it != allCluster.end(); )
            {
                // cout << (*it)->head->feat->index << endl;
                if( *it == n.a || *it == n.b )
                {
                    it = allCluster.erase( it );
                    // cout << "erase" << (*it)->head->feat->index << endl;
                }
                else
                    it++;
            }

            cluster* c = merge( n.a, n.b );
            for( int i = 0; i < allCluster.size(); i++ )
            {
                cluster* a = allCluster[i];
                double affi = affinity( a, c, w );
                affinityNode n( a, c, affi );
                Q.push( n );
            }

            allCluster.push_back(c);



        }




    }

}


void initialCluster( vector<feature>& feats, vector<cluster*>& allCluster )
{
    /* initial cluster with feature-graph
     * use BFS to find weak connected component in the graph,
     * each WCC is a cluster
     */
    int numOfFeats = feats.size();
    Mat allWeight;
    allWeight.create( numOfFeats, numOfFeats, CV_64FC1 );


    double* weight = (double*)allWeight.data;
    double s; // useless

    calcAllPairDist( weight, feats, &s, 2 );
    // print( allWeight );

    Mat trans = allWeight.t();
    // print( trans );

    Mat m = trans | allWeight;
    // print( m );

    bool* visited = new bool[ numOfFeats ];
    memset( visited, 0, numOfFeats*sizeof(bool) );

    for( int i = 0; i < numOfFeats; i++ )
    {
        if( !visited[i] )
        {
            cluster* newCluster = new cluster();
            BFS( i, newCluster, feats, visited, (double*)m.data );
            allCluster.push_back( newCluster );
        }
    }
    print( allCluster );


}

void BFS( int start, cluster* c, vector<feature>& feats, bool* visited, double* w )
{
    int numOfFeats = feats.size();
    queue<int> Q;
    int f;
    Q.push( start );
    while( !Q.empty() )
    {
        f = Q.front();
        Q.pop();
        if( !visited[f] )
        {
            c->insert( new clusterNode( &feats[f] ) );
            visited[f] = true;
            for( int i = 0; i < numOfFeats; i++ )
            {
                if( !visited[i] && (w[f*numOfFeats+i] != 0) )
                    Q.push(i);
            }
            // cout << Q.size();
        }
    }
}

void calcAllPairDist( double* w, vector<feature>& f, double* s, int k_KNN, vector< vector<sortAssistant> >& KNN_array )
{
    /* ( not the final weights )
     * w_ij = dist(i,j)^2, if j in KNN of i;
     *      = 0, otherwise.
     */
    int n = f.size();
    sortAssistant* neighbor = new sortAssistant[n];
    for( int i = 0; i < n; i++ )
    {
#pragma omp parallel for
        for( int j = 0; j < n; j++ )
        {
            neighbor[j].im_index = f[j].im_index;
            neighbor[j].pos_index = f[j].pos_index;
            neighbor[j].index = j;
            neighbor[j].dist = getEuclidDistSquare( f[i], f[j] );
            // file << getEuclidDistSquare( f[i], f[j] ) << " ";
        }

        std::sort( neighbor, neighbor+n, cmp );

        vector<sortAssistant> vec;
        KNN_array.push_back( vec );
        for( int j = 0; j < n; j++ )
        {
            // file << neighbor[j].dist << " ";
            if( j < k_KNN + 1 )
            {
                *s += neighbor[j].dist;
                KNN_array[i].push_back( neighbor[j] );
            }
            else
                neighbor[j].dist = 0.0;

            w[n*i+neighbor[j].index] = neighbor[j].dist;

        }
        // file << endl;
        // file << endl;

    }
}

void calcAllPairDist( double* w, vector<feature>& f, double* s, int k_KNN )
{
    /* ( not the final weights )
     * w_ij = dist(i,j)^2, if j in KNN of i;
     *      = 0, otherwise.
     */
    int n = f.size();
    sortAssistant* neighbor = new sortAssistant[n];
    for( int i = 0; i < n; i++ )
    {
#pragma omp parallel for
        for( int j = 0; j < n; j++ )
        {
            neighbor[j].index = j;
            neighbor[j].dist = getEuclidDistSquare( f[i], f[j] );
            // file << getEuclidDistSquare( f[i], f[j] ) << " ";
        }

        std::sort( neighbor, neighbor+n, cmp );

        for( int j = 0; j < n; j++ )
        {
            // file << neighbor[j].dist << " ";
            if( j < k_KNN + 1 )
            {
                *s += neighbor[j].dist;
            }
            else
                neighbor[j].dist = 0.0;
            w[n*i+neighbor[j].index] = neighbor[j].dist;

        }
        // file << endl;
        // file << endl;

    }
}

void calcWeight( double* w, int n, int KNN, double* s )
{
    /* w_ij is final weight of edge(i,j)
     * w_ij = exp( - dist(i,j)^2 / sigma^2 ), if j in KNN of i;
     *      = 0, otherwise.
     */
    *s = (*s) / n / KNN;
    for( int i = 0; i < n; i++ )
    {
        for( int j = 0; j < n; j++ )
        {
            double* weigh = w + n*i+j;
            if( *weigh == 0.0 )
                continue;
            else
                *weigh = exp( 0 - *weigh / *s );

        }
    }

}

int* getCluster( cluster* c )
{

    int* f = new int[c->count];
    clusterNode* n = c->head;
    for( int i = 0; i < c->count; i++ )
    {
        f[i] = n->feat->index;
        n = n->next;
    }
    return f;
}

cluster* merge( cluster* a, cluster* b )
{
    // merge two linked list and add up the counter
    cluster* c = new cluster();
    c->count = a->count + b->count;

    if( a->count == 0 )
    {
        c->head = b->head;
        return c;
    }
    if( b->count == 0 )
    {
        c->head = a->head;
        return c;
    }

    clusterNode* tail = a->head;
    while( tail->next != NULL )
            tail = tail->next;
    tail->next = b->head;
    c->head = a->head;
    return c;
}

double affinity( cluster* a, cluster* b, Mat& w )
{
    // matrix-operation to get affinity b/w two clusters
    Mat Wab;
    Mat Wba;
    int* lista = getCluster( a );
    int* listb = getCluster( b );
    int Na = a->count;
    int Nb = b->count;
    Wab.create( Na, Nb, CV_64FC1 );
    Wba.create( Nb, Na, CV_64FC1 );

    for( int i = 0; i < Na; i++ )
    {
        int row = lista[i];
        for( int j = 0; j < Nb; j++ )
        {
            Wab.at<double>(i,j) = w.at<double>(row,listb[j]);
        }
    }

    for( int i = 0; i < Nb; i++ )
    {
        int row = listb[i];
        for( int j = 0; j < Na; j++ )
        {
            Wba.at<double>(i,j) = w.at<double>(row,lista[j]);
        }
    }
    Mat ones_a( Na, 1, CV_64FC1, Scalar(1.0) );
    Mat ones_b( Nb, 1, CV_64FC1, Scalar(1.0) );

    Mat m1 = ones_a.t() * Wab * Wba * ones_a;
    Mat m2 = ones_b.t() * Wba * Wab * ones_b;

    // return 0.0;
    return ( m1.at<double>(0,0) + m2.at<double>(0,0) )/ ((Na+Nb)*(Na+Nb));

}

void printMbyN( double* mat, int m, int n )
{
    for( int i = 0; i < m; i++ )
    {
        for( int j = 0; j < n; j++ )
            file << mat[n*i+j] << "\t";
        file << std::endl;
    }
    file << std::endl;
}

/*void test()
{
    ifstream file("//home//yyli//桌面//cluster//R15");
    double f1, f2;
    int c;
    int n = 0;
    vector<feature> f;
    while( file >> f1 >> f2 >> c )
    {
        feature feat(f1,f2,c);
        f.push_back(feat);

    }


    clustering( f );


}*/

void debug()
{
    cout << "here!" << endl;
}

void print( Mat& mt )
{
    int m = mt.rows;
    int n = mt.cols;
    printMbyN( (double*)mt.data, m, n );
}

void print( vector<cluster*>& allCluster )
{
    /*
    for( vector<cluster*>::iterator it = allCluster.begin(); it != allCluster.end(); it++ )
    {
        for( clusterNode* n = (*it)->head; n != NULL; n = n->next )
            n->feat->print();
        cout << endl;
    }
    */
}

/*
int main()
{
    test();

    return 0;
}
*/





