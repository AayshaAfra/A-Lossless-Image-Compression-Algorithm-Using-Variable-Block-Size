clc
clear all
close all

% Read the input image
ori_image = imread("E:\download.jfif");
image = rgb2gray(ori_image);
figure,imshow(image);
% Initial block size
initialBlockSize = 8;
codeword=[];
header=[];
CompressedImage=cell(1, 0);
maxOffset = 50; minOffset = -50;
 
% Threshold for base-offset coding
threshold = 48; % Adjust the threshold value based on your requirement

% Iterate over each block size starting from initialBlockSize
blockSize = initialBlockSize;
while blockSize >= 2
    % Partition the image into blocks
    blocks = im2col(image, [blockSize, blockSize], 'distinct');
    
    % Iterate over each block
    for i = 1:size(blocks, 2)
        block = blocks(:, i);
        
        maxDifference = max(abs(diff(block)));
        
        offsetRange = calculateOffsetRange(block);
        if all(block == block(1))
            % Encode using run-length coding
            codeword = runLengthCoding(block);
            category = findCategory(blockSize,[0 0],'run-length');
        elseif(blockSize==2)
            codeword=blockMatchingCoding(block,8);
             category = 5;
        elseif(maxOffset<=threshold && minOffset>= -threshold)
            codeword = baseOffsetCoding(block);
            category = findCategory(blockSize, offsetRange, 'base-offset');
        else
            codeword = block;
            category = 27;
        end
        CompressedImage =[CompressedImage,codeword];
        % Write the category information to the header file
        createHeaderFile(codeword);
         header = [header; category];
     end
    
    % Update the block size for the next iteration
    blockSize = blockSize / 2;
end

% Apply Huffman coding to the header file
 compressHeaderFile(header);
    
 % Save the compressed image file
 saveCompressedImage(CompressedImage);
     
     [rows, cols, ~] = size(image);
     [rows1, cols1, ~] = size(CompressedImage);
     
     originalImageSize=rows*cols;
     compressedImageSize=rows1*cols1;
   %calculate Compress Efficiency
   calculateCompressionEfficiency(originalImageSize, compressedImageSize);
      
      
      
    function compressionEfficiency = calculateCompressionEfficiency(originalImageSize, compressedImageSize)
    % Calculate the compression efficiency
    compressionEfficiency = ((originalImageSize - compressedImageSize) / originalImageSize) * 100;
     fprintf('The original image size is %d\n', originalImageSize );
     fprintf('The compressed image size is %d\n', compressedImageSize);
    fprintf('The compression efficiency is %d\n', compressionEfficiency);
    end

% Implement the saveCompressedImage function to save the compressed image
function saveCompressedImage(CompressedImage)
    % Convert the cell array to a string
   compressedStrings = cellfun(@num2str, CompressedImage, 'UniformOutput', false);
    compressedString = sprintf('%s', compressedStrings{:});
    % Write the compressed string to a binary file
    fileID = fopen('C:\Users\user\Desktop\mm\compressed_image.bin', 'w');
    fwrite(fileID, compressedString, 'ubit1');
    fclose(fileID);
end

function binary = decimalToBinary(decimal, numBits)
    binary = zeros(numel(decimal), numBits);
    for i = 1:numel(decimal)
        binary(i, :) = bitget(decimal(i), numBits:-1:1);
    end
end

function codeword = baseOffsetCoding(block)
    % Encode using base-offset coding
    blockSize = size(block, 1);
    numPixels = blockSize^2;
    
    % Calculate the maximum offset and direction of the base pixel
    [offsets, direction] = calculateOffsetsAndDirection(block);
    
    disp(offsets);
    
    % Convert offsets to binary representation
    maxOffset = max(offsets);
    disp(maxOffset);
    if maxOffset > 0
        numBits = ceil(log2(double(maxOffset)));
        offsetBits = decimalToBinary(offsets, numBits);
    else
        offsetBits = zeros(size(offsets));  % No bits needed if max(offsets) is zero
    end
    
    % Convert direction to binary representation
    directionBits = decimalToBinary(direction, 2);
    
    % Concatenate offset bits and direction bits to form the codeword
    codeword = [offsetBits(:)', directionBits];
end

function [offsets, direction] = calculateOffsetsAndDirection(block)
    % Calculate offsets for all pixels in the block and direction of the base pixel
    basePixel = block(1);
    offsets = block - basePixel;
    
    % Determine the direction of the base pixel based on nearest neighborhood principle
    % (implementation of the nearest neighborhood principle)
    direction = determineDirection(block);
end

function direction = determineDirection(block)
    % Determine the direction based on the nearest neighborhood principle
    [row, col] = size(block);
    
    % Calculate the indices of neighboring pixels
    north = max(1, row - 1);
    east = min(col + 1, col);
    northeast = sub2ind([row, col], north, east);
    
    % Determine the direction based on the average of the neighboring pixels
    average = mean([block(north), block(east), block(northeast)]);
    [~, direction] = min(abs([block(north), block(east), block(northeast)] - average));
end

function codeword = blockMatchingCoding(block, searchSpace)
    % Encode using block matching coding
    blockPattern = block;
    blockSize = size(block, 1);
    
    % Search for an identical pattern within the search space
    [row, col] = findPattern(blockPattern, searchSpace);
    
    if isempty(row) || isempty(col)
        % Identical pattern not found, encode using base-offset coding
        codeword = baseOffsetCoding(block);
    else
        % Identical pattern found, encode differences along z and y directions
        differences = blockPattern - searchSpace(row:row+blockSize-1, col:col+blockSize-1);
        identicalPattern = 1;  % Flag to indicate identical pattern found
        codeword = [differences(:)', identicalPattern];
    end
end

function [row, col] = findPattern(pattern, searchSpace)
    % Search for an identical pattern within the search space
    [M, N] = size(searchSpace);
    [m, n] = size(pattern);
    
    for row = 1 : M - m + 1
        for col = 1 : N - n + 1
            if isequal(pattern, searchSpace(row:row+m-1, col:col+n-1))
                % Identical pattern found
                return;
            end
        end
    end
    
    % Identical pattern not found
    row = [];
    col = [];
end

function codeword = runLengthCoding(block)
    % Encode using run-length coding
    basePixel = block(1);
    blockLength = numel(block);
    codeword = [basePixel, blockLength];
end

function category = findCategory(blockSize, offsetRange, encoding)
    % Define the category table
    categoryTable = [
        0   8   [0   0]     "run-length"       2
        1   4   [0   0]     "run-length"       2
        2   2   [0   0]     "run-length"       2
        3   8   [1   2]     "base-offset"      66
        4   4   [1   2]     "base-offset"      18
        5   2   [0   0]     "block matching"   8
        6   2   [-1  0]     "base-offset"      10
        7   2   [0   1]     "base-offset"      10
        8   2   [-2  1]     "base-offset"      10
        9   2   [0   3]     "base-offset"      10
        10  2   [-3  0]     "base-offset"      10
        11  2   [-3  4]     "base-offset"      14
        12  2   [-5  2]     "base-offset"      14
        13  2   [-2  5]     "base-offset"      14
        14  2   [-7  8]     "base-offset"      18
        15  2   [-4  11]    "base-offset"      18
        16  2   [-11 4]     "base-offset"      18
        17  2   [-2  13]    "base-offset"      18
        18  2   [-13 2]     "base-offset"      22
        19  2   [-15 16]    "base-offset"      22
        20  2   [-4  27]    "base-offset"      22
        21  2   [-27 4]     "base-offset"      22
        22  2   [-20 11]    "base-offset"      22
        23  2   [-11 20]    "base-offset"      22
        24  2   [-31 32]    "base-offset"      26
        25  2   [-15 48]    "base-offset"      26
        26  2   [-48 15]    "base-offset"      26
        27  0   [0   0]     "raw data"         32
    ];
     
      % Convert blockSize to a string
    blockSizeStr = num2str(blockSize);
    
    % Convert offsetRange to a cell array of strings
    offsetRangeStr = cellstr(num2str(offsetRange));
    
    % Find the matching category based on blockSize, offsetRange, and encoding
    match = find(all([strcmp(categoryTable(:, 2), blockSizeStr), ismember(categoryTable(:, 3), offsetRangeStr)], 2) & strcmp(categoryTable(:, 4), encoding));
    
    % If no match is found, set the category to -1
    if isempty(match)
        category = -1;
    else
        category = categoryTable(match, 1);
    end
end


function createHeaderFile(codeword)
    % Open the header file for writing
    headerFile = fopen('C:\Users\user\Desktop\mm\header.txt', 'w');
   
    % Check if the file was opened successfully
    if headerFile == -1
        error('Failed to open header file.');
    end
    
    % Get the category and frequency from the codeword
    category = codeword(1);
    freq = codeword(2);
    
    % Write the category and frequency to the header file
    fprintf(headerFile, 'Category: %d, Frequency: %d\n', category, freq);
    
    % Close the header file
    fclose(headerFile);
end

function compressHeaderFile(headerText)
     % Apply Huffman coding to the header
symbols = unique(headerText);
counts = histc(headerText(:), symbols);
probabilities = counts / sum(counts); % Normalize the counts to obtain probabilities

[dict, avglen] = huffmandict(symbols, probabilities);
compressedHeader = huffmanenco(headerText(:), dict);
    
    % Write the compressed header to a file
    compressedHeaderFile = fopen('C:\Users\user\Desktop\mm\compressed_header.bin', 'w');
     if compressedHeaderFile == -1
        error('Failed to open compressed header file.');
    end
    fwrite(compressedHeaderFile, compressedHeader, 'ubit1');
    fclose(compressedHeaderFile);
end
function offsetRange = calculateOffsetRange(block)
    % Calculate the offset range for a block
    
    % Get the neighboring pixels
    northPixel = block(1);
    eastPixel = block(end);
    northEastPixel = block(end-1);

    % Calculate the average of north and east pixels
    avgPixel = round((northPixel + eastPixel) / 2);

    % Calculate the offsets from each neighboring pixel
    offsets = [block - northPixel; block - eastPixel; block - northEastPixel; block - avgPixel];

    % Calculate the maximum and minimum offsets
    maxOffset = max(offsets);
    minOffset = min(offsets);

    % Set the offset range
    offsetRange = [minOffset, maxOffset];
end
