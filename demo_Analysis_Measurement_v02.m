% demo show measurement
clear
for netId = [1 2 3]
network = ['DoC-DCS' num2str(netId)];
%  network = 'CSNet'; 
for subRate = [0.1 0.2 0.3]
    modelName   = [network '_r' num2str(subRate)];
    
    folderTest  = 'Classic13_512';
    ext         =  {'*.jpg','*.png','*.bmp', '*.pgm', '*.tif'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})) );
    end
    
    for i = 1:1:1%numel(filePaths)
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        data = load(['histResults\' network '\' nameCur '_r' num2str(subRate) '.mat' ]);
       
       
        [m, n, c] = size(data.Y); 
        rand_c    = randperm(c-1, floor(c/2)); 
        rand_m    = randperm(m-1, floor(m/2)); 
        rand_n    = randperm(n-1, floor(n/2)); 
        
        count     = 0; 
        for i1 = 1:1:length(rand_c)
            for i2 = 1:1:length(rand_m)                
                for i3 = 1:1:length(rand_n)
                    count = count + 1; 
                    x(count)  = data.Y(rand_m(i3), rand_n(i2), rand_c(i1)); 
                    x_c(count) = data.Y(rand_m(i3), rand_n(i2), rand_c(i1) + 1); 
                    x_m(count) = data.Y(rand_m(i3)+1, rand_n(i2), rand_c(i1)); 
                    x_n(count) = data.Y(rand_m(i3), rand_n(i2) + 1, rand_c(i1)); 
                end
            end
        end
        
        % show correlation;
        if ~exist(['Results_Corr\' network '\']), mkdir(['Results_Corr\' network '\']); end
        figure(1); scatter(x, x_c); axis([-2 2 -2 2]),set(gca, 'fontsize', 18); grid on;
        im_c = export_fig;
        figure(2); scatter(x, x_m); axis([-2 2 -2 2]), set(gca, 'fontsize', 18); grid on;
        im_m = export_fig; 
        figure(3); scatter(x, x_n); axis([-2 2 -2 2]), set(gca, 'fontsize', 18); grid on; 
        im_n = export_fig; 
        imwrite(im_c, ['Results_Corr\' network '\' nameCur '_r' num2str(subRate) '_c.png']); 
        imwrite(im_m, ['Results_Corr\' network '\' nameCur '_r' num2str(subRate) '_m.png']); 
        imwrite(im_n, ['Results_Corr\' network '\' nameCur '_r' num2str(subRate) '_n.png']); 
        
    end
    
    
end
end