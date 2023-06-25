// Set the total number of channels here
var totalChannels = 60;

// Loop over each channel
for (var channel = 1; channel <= totalChannels; channel++) {
    run("Bio-Formats", "open=Z:/ny1/IBEX/Thymus/Sample_13/Thy55_Cycle1_and_Cycle9_Final_Upload.ims color_mode=Default rois_import=[ROI manager] specify_range view=Hyperstack stack_order=XYCZT series_1 c_begin_1=" + channel + " c_end_1=" + channel + " c_step_1=1");
    run("Z Project...", "projection=[Max Intensity]");
    
    // Pad the channel number with zeroes to maintain file sorting
    var paddedChannel = "" + channel;
    while (paddedChannel.length < 2) {
        paddedChannel = "0" + paddedChannel;
    }

    // Save the result with the appropriate file name
    saveAs("Tiff", "Z:/ny1/IBEX/Thymus/Sample_13/series/Thy55_Cycle1_and_Cycle9_Final_Upload.ims_ch" + paddedChannel + "_max.tif");
    
    // Close the images
    close();
    close();
}