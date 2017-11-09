%% Get images folder on which the RCPR is to be performed.

% Change folder path to the image directory to use.
folder = 'F:\Installer Backup\rcpr_v1\rcpr_v1_stable\video_tracking\images'

% Search for jpg files
filePattern = fullfile(folder, '*.jpg');

f=dir(filePattern)
files={f.name}

% Read all images and store them in cell array
for k=1:numel(files)
	fullFileName = fullfile(folder, files{k});
	cellArrayOfImages{k}=imread(fullFileName);
end

% Convert cell array to 3D array
s.Is = cat(4, cellArrayOfImages{:});

% Define the info of the sequence file to be exported
info=struct('codec','jpg','fps',1);

% Convert the images to a sequence file that can be fed to the trained RCP network
seqIo( 'sample3.seq', 'frImgs',info, s );
