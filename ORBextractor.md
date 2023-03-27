# ORB feature extractor module
- This module is related to the ORB feature extraction.

- if other parts of the SLAM system want to use ORB feature extractor module, they need to call ORBextractor::operator() function.
```c++
oid ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{ 
    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    vector < vector<KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);

    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    // kind of dangerous
    _keypoints.clear();
    // allocate space for efficiency
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        Mat workingMat = mvImagePyramid[level].clone();

        // GaussianBlur
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}
```
- Thus I will start from this function, and figure out the whole process of ORB extractor module. Keep me up.

- After reading aforementioned code, we can tell that the program acquired an Image by input parameter `_image`, then the programm will call function `ComputePyramid(image)`. Now, let's move our attention to this function. 
```c++
void ORBextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);            
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);            
        }
    }

}
```

## What did the function `ComputePyramid(cv::Mat image)` do?
- It is used for resizing original image to different shape by parameter `scale`, and save different level images to vector  `mvImagePyramid`. Suppose the scale = 1.2, the vector `mvImagePyramid[0]` saves image with size(original\_height / scale^0^, original\_width / scale^0^), and the vector`mvImagePyramid[i]` stores image with size (original\_height / scale^i^, original\_width / scale^i^). Moreover, every resizeing factor is saved in vector `mvInvScaleFactor[level]`, if you want to know how to initialize it, you can go to check the `construct function of ORBextrctor`. What's more, I feel like this part of code can be improved, because the variables **EDGE_THRESHOLD, wholeSize, temp, masktemp**, and the function `copyMakeBorder(), Rect()` has nothing to do with **resized image**. You can try different values of variable **EDGE_THRESHOLD**, it won't affect the shape of resized image.

Now, we know what the function `ComputePyramid(cv::Mat image)` did, and we need to keep moving. According to the code, we found that it called the function `ComputeKeyPointsOctTree`. Let's figure out what this function did !
```c++
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30;

    for (int level = 0; level < nlevels; ++level)
    {
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures*10); // why ?

        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        const int nCols = width/W; // compute the number of window (Cell)
        const int nRows = height/W;
        const int wCell = ceil(width/nCols); //Compute the width of each Cell
        const int hCell = ceil(height/nRows); //Compute the height of each Cell

        for(int i=0; i<nRows; i++)
        {
            const float iniY =minBorderY+i*hCell;
            float maxY = iniY+hCell+6;

            if(iniY>=maxBorderY-6)
                continue;
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            for(int j=0; j<nCols; j++)
            {
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
                if(iniX>=maxBorderX-6)
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                     vKeysCell,iniThFAST,true);

                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,minThFAST,true);
                }

                if(!vKeysCell.empty())
                {
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        (*vit).pt.x+=j*wCell; 
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}
```
## what did function `ComputeKeyPointsOctTree()` do?
- Just by observation, we can infer it may has relationship with `KeyPoints`, actually, it does. Let me just told you the answer after I analyzed. Firstly, it will extract keypoints in `a rectangle area [ left-top point:(EDGE_THRESHOLD-3, EDGE_THRESHOLD -3), right-bottom point:(img.rows-EDGE_THRESHOLD+3, img.cols-EDGE-THRESHOLD+3)]` in `PyramidImages` by `FAST` feature extractor (implemented by opencv). Besides, the type of `FAST` is `TYPE_9_16`, 9 denotes N, 16 denotes number of surronding pixels ( if you have no idea what FAST is, you can go to [here](https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html)). Before output keypoints in different level image, It will select keypoints again by call function `DistributeOctTree()`.

```c++
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes   
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    //allocate all keypoints to the ExtractorNode
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4); //hypothesize every single ExtractorNode will divide into 4

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);                    
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }       

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;

}
```
## What did the function `ComputeKeyPointsOctTree()` do?
- Although it's long, don't afraid of it. Just let me explain what this function did. Before calling this function, we got lots of keypoints by FAST feature extractor already. But they are in an uneven distribution. Thus, the target of this function is to select more scattered keypoints. According to [ORB paper](https://ieeexplore.ieee.org/document/6126544), farer two keypoints are, less correlated they are. And these keypoints can reprensent the image feature better. Back to the code, how did this function make the keypoints scattered? `Firstly, the whole area with keypoints will divide into four blocks, then each block would keep dividing if it satisfies certain conditions (more than one keypoints in this block) untile the number of whole blocks aboves or equals N (assigned parameter). Secondly, if we got enough blocks, the function would pick up biggest response keypoints in each block. There selected keypoints are final keypoints, these operations will happen to every level image`. Thus, after executing function `DistributeOctTree()` on every level image, we acquired all final keypoints that stored in the vector `allKeypoints[level]`. The program will call back to function`ComputeKeyPointsOctTree()`.Then, it will call function`computeOrientation()` to compute the orientation for each final keypoints.
```c++
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}
```

```c++
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}
```

## What did the function `computeOrientation()` and function `IC_Angle()` do?
- The function `computeOrientation()` is easy to understand. Just a loop on each key point to call function`IC_Angle()` compute its orientation. Thus, Let's pay more attention to function `IC_Angle()`. According to [ORB paper](https://ieeexplore.ieee.org/document/6126544), the orientation of a keypoint is treated as an itensity centroid offset to the keypoint in a circle patch area. Before compute the orientation, we need to compute `m_10` and `m_01`. But how? The logic of the function is like this. Making a circle patch around a keypoint. Firstly, it will compute `m_10` for pixels in center line. The value of y is zero in center line, so, `m_01` is zero in the center line. And then, proceed over two lines at the same time. The variable v stands for y, the variable u stands for x. Finally, return a degree by execute function `fastAtan2(m_01, m1_0)`, this degree is a keypoint's orientation.
`
>We started from the function `ORBextractor::operator()`, after the program executed function`ComputeKeyPointsOctTree()`, the program extracted oFast feature keypoints in every level image. But it is no over, because the function`ORBextractor::operator()` has no been done. Let's keep moving.


## Computing descriptors, focus on function`computeDescriptors()` and function `computeOrbDescriptor()`
- Those two function all has relationship with the descriptors, but same to aforementioned function `computeOrientation()` and function `IC_Angle()`. The key function is later, and former is easier to understand. The parameter `pattern` is realted to selected 2\*N points for generating desriptor process. More specifically, it stores all locations of 2\*N points. `GET_VALUE(idx)` denotes the itensity of one selected point. `I guess the program can be improved by redefine the type of pattern, because it is static type currently, and it still take sizeof(int)*1024 memory after the program generates descriptors of all final keypoints`.
- More specifically, 
```c++
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}
```
```c++

static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]


    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}
```
