// Global variables for file path, channel and Z plane numbers
var ims_file;
var channels;
var z_planes;

function process_ims() {
    // Get the base name of the file without extension for saving results
    var baseNameIndex = ims_file.lastIndexOf("/");
    var baseName;
    if (baseNameIndex != -1) {
        baseName = ims_file.substring(baseNameIndex + 1, ims_file.length());
    } else {
        baseName = ims_file;
    }
    
    // Remove extension from the baseName
    var extensionIndex = baseName.lastIndexOf(".");
    if (extensionIndex != -1) {
        baseName = baseName.substring(0, extensionIndex);
    }

    // Get the directory of the file for saving results
    var dirName = File.getParent(ims_file);

    // Make sure there is a folder named 'series_multiplane' in the directory
    var resultDir = dirName + "/series_multiplane";
    if (!File.exists(resultDir)) {
        File.makeDirectory(resultDir);
    }

    for (var channel = 1; channel <= channels; channel++) {
        // Open all Z planes in the current channel
        run("Bio-Formats", "open=" + ims_file + " color_mode=Default rois_import=[ROI manager] specify_range view=Hyperstack stack_order=XYCZT series_1 c_begin_1=" + channel + " c_end_1=" + channel + " c_step_1=1 z_begin_1=1 z_end_1=" + z_planes + " z_step_1=1");
         // Pad the channel number and Z plane number with zeroes to maintain file sorting
           
		run("Image Sequence... ", " dir="+resultDir+"/ format=TIFF name="+baseName + "_ch" + IJ.pad(channel, 2) + "_z"+" start=1 digits=2");

        // Close the images
        close();
    }
}



// Open the CSV file
var filename = "Z:/ny1/IBEX/ibex_folder_batch.csv";

var fileContent = File.openAsString(filename);

// Split the content into lines
var lines = split(fileContent, "\n");

// Skip the header line
var lines = Array.slice(lines, 1);

for (var i = 0; i < lines.length; i++) {
    var line = lines[i];

    // Split the line into components
    components = split(line, ",");

    // Get the ims file and the channel and Z plane numbers from the components
    ims_file = components[0];
    channels = parseInt(components[1]);
    z_planes = parseInt(components[2]);

    // Process the ims file with the given channel and Z plane numbers
    process_ims();
}
