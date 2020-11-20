%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @article{Canh2018_MSCSNet,
%   title={Multi-Scale Deep Compressive Sensing Network},
%   author={Thuong, Nguyen Canh and Byeungwoo, Jeon},
%   conference={IEEE International Conference on Visual Comunication and Image Processing},
%   year={2018}
% }
% by Thuong Nguyen Canh (9/2018)
% ngcthuong@gmail.com
% https://github.com/AtenaKid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You need to install Matconvnet in order to run this code 

warning('off','all')
addpath('D:\matconvnet-1.0-beta25\matlab\mex');
addpath('D:\matconvnet-1.0-beta25\matlab\simplenn');
% addpath('D:\matconvnet-1.0-beta25\matlab');

addpath('.\utilities');

folderTest  = 'Test_IBug';

networkTest = {'CSNet' 'W-DCS1' 'W-DCS2' 'W-DCS3' 'SS-DCS1' 'SS-DCS2' 'SS-DCS3' ...
               'P-DCS1' 'P-DCS2' 'P-DCS3' 'DoC-DCS1' 'DoC-DCS2' 'DoC-DCS3'};      % 10

showResult  = 0;
writeRecon  = 1;
featureSize = 64;
blkSize     = 32; 
isLearnMtx  = [1, 0];
network     = networkTest{1}; 

for samplingRate = [0.1:0.1:0.3]
    modelName   = [network '_r' num2str(samplingRate)]; %%% model name
        
    data = load(fullfile('models', network ,[modelName,'.mat']));
    net  = dagnn.DagNN.loadobj(data.net);
    if strcmp(network,'CSNet')
        net.renameVar('x0', 'input'); 
        net.renameVar('x12', 'prediction'); 
    else
        net.removeLayer(net.layers(end).name) ;
    end
        
    net.mode = 'test';
    net.move('gpu');
        
    %%% read images
    ext         =  {'*.jpg'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile('F:\DissertationCode\Face_DataSet',folderTest,ext{i})) );
    end
    
    
    count = 1;
    allName = cell(1);
    
    for i = 1:length(filePaths)
        
        %%% read images
        image0 = imread(fullfile('F:\DissertationCode\Face_DataSet', folderTest, filePaths(i).name));
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        allName{count} = nameCur;
        output = zeros(size(image0)); 
        
        for c = 1:1:3
            image = image0(:,:,c);        
            label = im2single(image);

            input = label;
            input = gpuArray(input);
            tic
            net.eval({'input', input}) ;
            time(i) = toc; 
            out1 = net.getVarIndex('prediction') ;
            output(:, :, c) = gather(squeeze(gather(net.vars(out1).value)));            
        end
        
        if writeRecon
            folder  = ['Results\' folderTest '\2Image_' network '\subrate' num2str(samplingRate)]; 
            if ~exist(folder), mkdir(folder); end
            fileName = [folder '\'  allName{count} '_subrate' num2str(samplingRate) '.png'];
            imwrite(im2uint8(output), fileName );            
            count = count + 1;
        end        
    end
end
