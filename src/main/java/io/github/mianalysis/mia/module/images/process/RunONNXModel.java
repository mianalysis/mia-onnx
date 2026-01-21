package io.github.mianalysis.mia.module.images.process;

import java.util.HashMap;

import org.scijava.Priority;
import org.scijava.plugin.Plugin;

import com.drew.lang.annotations.Nullable;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.TensorInfo;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import io.github.mianalysis.mia.MIA;
import io.github.mianalysis.mia.module.AvailableModules;
import io.github.mianalysis.mia.module.Categories;
import io.github.mianalysis.mia.module.Category;
import io.github.mianalysis.mia.module.Module;
import io.github.mianalysis.mia.module.Modules;
import io.github.mianalysis.mia.module.images.transform.ExtractSubstack;
import io.github.mianalysis.mia.module.objects.measure.miscellaneous.ApplyWekaObjectClassification;
import io.github.mianalysis.mia.object.Workspace;
import io.github.mianalysis.mia.object.image.Image;
import io.github.mianalysis.mia.object.image.ImageFactory;
import io.github.mianalysis.mia.object.parameters.BooleanP;
import io.github.mianalysis.mia.object.parameters.ChoiceP;
import io.github.mianalysis.mia.object.parameters.FilePathP;
import io.github.mianalysis.mia.object.parameters.InputImageP;
import io.github.mianalysis.mia.object.parameters.OutputImageP;
import io.github.mianalysis.mia.object.parameters.ParameterState;
import io.github.mianalysis.mia.object.parameters.Parameters;
import io.github.mianalysis.mia.object.parameters.SeparatorP;
import io.github.mianalysis.mia.object.parameters.text.IntegerP;
import io.github.mianalysis.mia.object.parameters.text.MessageP;
import io.github.mianalysis.mia.object.parameters.text.StringP;
import io.github.mianalysis.mia.object.refs.collections.ImageMeasurementRefs;
import io.github.mianalysis.mia.object.refs.collections.MetadataRefs;
import io.github.mianalysis.mia.object.refs.collections.ObjMeasurementRefs;
import io.github.mianalysis.mia.object.refs.collections.ObjMetadataRefs;
import io.github.mianalysis.mia.object.refs.collections.ParentChildRefs;
import io.github.mianalysis.mia.object.refs.collections.PartnerRefs;
import io.github.mianalysis.mia.object.system.Status;
import io.github.mianalysis.mia.process.imagej.ImageTiler;
import net.imagej.ImageJ;
import net.imagej.patcher.LegacyInjector;

@Plugin(type = Module.class, priority = Priority.LOW, visible = true)
public class RunONNXModel extends Module {

    public static final String INPUT_SEPARATOR = "Image input";

    public static final String INPUT_IMAGE = "Input image";

    public static final String INPUT_NOTE = "Input note";

    public static final String OUTPUT_SEPARATOR = "Image output";

    public static final String OUTPUT_IMAGE = "Output image";

    /**
     * By default images will be saved as floating point 32-bit (probabilities in
     * the range 0-1); however, they can be converted to 8-bit (probabilities in the
     * range 0-255) or 16-bit (probabilities in the range 0-65535). This is useful
     * for saving memory or if the output probability map will be passed to image
     * threshold module.
     */
    public static final String OUTPUT_BIT_DEPTH = "Output bit depth";

    public static final String OUTPUT_SUBSET_OF_CLASSES = "Output subset of classes";

    public static final String OUTPUT_CLASSES = "Output classes";

    public static final String MODEL_SEPARATOR = "Model controls";

    public static final String MODEL_PATH = "Model path";

    public static final String TILE_OVERLAP = "Tile overlap (px)";

    public interface OutputBitDepths {
        String EIGHT = "8";
        String SIXTEEN = "16";
        String THIRTY_TWO = "32";

        String[] ALL = new String[] { EIGHT, SIXTEEN, THIRTY_TWO };

    }

    public static void main(String[] args) {
        // The following must be called before initialising ImageJ
        LegacyInjector.preinit();

        // Creating a new instance of ImageJ
        new ij.ImageJ();

        // Launching MIA
        new ImageJ().command().run("io.github.mianalysis.mia.MIA_", false);

        // Adding the current module to MIA's list of available modules.
        AvailableModules.addModuleName(RunONNXModel.class);

    }

    public RunONNXModel(Modules modules) {
        super("Run ONNX model", modules);
    }

    @Override
    public String getVersionNumber() {
        return "1.0.0";
    }

    @Override
    public String getDescription() {
        return "";
    }

    @Override
    public Category getCategory() {
        return Categories.IMAGES_PROCESS;
    }

    public static Image tileImage(OrtSession session, Image inputImage, int tileOverlap) throws OrtException {
        ImagePlus inputIpl = inputImage.getImagePlus();

        TensorInfo inputInfo = (TensorInfo) session.getInputInfo().get("input").getInfo();
        long[] inputShape = inputInfo.getShape();
        int inputWidth = (int) inputShape[2];
        int inputHeight = (int) inputShape[3];

        if (inputIpl.getWidth() > inputWidth || inputIpl.getHeight() > inputHeight) {
            int xNumTiles = ImageTiler.getTileCount(inputIpl.getWidth(), tileOverlap, inputWidth);
            int yNumTiles = ImageTiler.getTileCount(inputIpl.getHeight(), tileOverlap, inputHeight);
            ImagePlus outputIpl = ImageTiler.tile(inputIpl, xNumTiles, yNumTiles, inputWidth, inputHeight, tileOverlap,
                    tileOverlap, ImageTiler.TileAxes.T);
            return ImageFactory.createImage(inputImage.getName(), outputIpl);

        } else if (inputIpl.getWidth() == inputWidth || inputIpl.getHeight() == inputHeight) {
            return inputImage;

        } else {
            MIA.log.writeWarning("Input image smaller than minimum size of width = " + inputWidth + "px and height = "
                    + inputHeight);

            return null;

        }
    }

    public static Image stitchImage(Image tiledImage, String outputImageName, int tileOverlap, int outputWidth,
            int outputHeight) {
        ImagePlus tiledIpl = tiledImage.getImagePlus();

        int tileWidth = tiledIpl.getWidth();
        int tileHeight = tiledIpl.getHeight();

        int xNumTiles = ImageTiler.getTileCount(outputWidth, tileOverlap, tileWidth);
        int yNumTiles = ImageTiler.getTileCount(outputHeight, tileOverlap, tileHeight);

        ImagePlus outputIpl = ImageTiler.stitch(tiledIpl, xNumTiles, yNumTiles, tileWidth, tileHeight, tileOverlap,
                tileOverlap, outputWidth, outputHeight, ImageTiler.TileAxes.T);

        return ImageFactory.createImage(outputImageName, outputIpl);

    }

    public static OnnxTensor getInputTensor(OrtEnvironment environment, OrtSession session, Image inputImage)
            throws OrtException {
        ImagePlus inputIpl = inputImage.getImagePlus();

        TensorInfo inputInfo = (TensorInfo) session.getInputInfo().get("input").getInfo();
        long[] inputShape = inputInfo.getShape();
        int inputChannels = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int inputHeight = (int) inputShape[3];

        float[][][][] floatArray = new float[1][inputChannels][inputWidth][inputHeight];
        for (int c = 0; c < inputChannels; c++) {
            inputIpl.setC(c + 1);
            ImageProcessor inputIpr = inputIpl.getProcessor();
            for (int x = 0; x < inputWidth; x++)
                for (int y = 0; y < inputHeight; y++)
                    floatArray[0][c][x][y] = inputIpr.getPixelValue(x, y);
        }

        return OnnxTensor.createTensor(environment, floatArray);

    }

    public static Image createEmptyOutputImage(OrtSession session, String outputImageName, Image inputImage,
            int bitDepth, @Nullable int[] classList) throws OrtException {
        ImagePlus ipl = inputImage.getImagePlus();

        TensorInfo outputInfo = (TensorInfo) session.getOutputInfo().get("target").getInfo();
        long[] outputShape = outputInfo.getShape();
        int nChannels = classList == null ? (int) outputShape[1] : classList.length;

        ImagePlus outputIpl = IJ.createHyperStack(outputImageName, ipl.getWidth(), ipl.getHeight(),
                nChannels, ipl.getNSlices(), ipl.getNFrames(), bitDepth);

        return ImageFactory.createImage(outputImageName, outputIpl);

    }

    public static Image getOutputImage(OrtSession session, Result outputMap, String outputImageName, int bitDepth,
            @Nullable int[] classList) throws OrtException {
        TensorInfo outputInfo = (TensorInfo) session.getOutputInfo().get("target").getInfo();
        long[] outputShape = outputInfo.getShape();
        int nChannels = classList == null ? (int) outputShape[1] : classList.length;
        int outputWidth = (int) outputShape[2];
        int outputHeight = (int) outputShape[3];

        float[][][][] output = (float[][][][]) outputMap.get(0).getValue();

        ImagePlus outputIpl = IJ.createHyperStack(outputImageName, outputWidth, outputHeight, nChannels, 1, 1,
                bitDepth);

        if (classList == null) {
            for (int c = 0; c < nChannels; c++) {
                outputIpl.setC(c + 1);
                ImageProcessor outputIpr = outputIpl.getProcessor();
                for (int x = 0; x < outputWidth; x++)
                    for (int y = 0; y < outputHeight; y++)
                        if (bitDepth == 8)
                            outputIpr.set(x, y, (int) Math.round(output[0][c][x][y] * 255));
                        else if (bitDepth == 16)
                            outputIpr.setf(x, y, (int) Math.round(output[0][c][x][y] * 65535));
                        else if (bitDepth == 32)
                            outputIpr.setf(x, y, output[0][c][x][y]);
            }
        } else {
            for (int cIdx = 0; cIdx < classList.length; cIdx++) {
                outputIpl.setC(cIdx + 1);
                int c = classList[cIdx];
                ImageProcessor outputIpr = outputIpl.getProcessor();
                for (int x = 0; x < outputWidth; x++)
                    for (int y = 0; y < outputHeight; y++)
                        if (bitDepth == 8)
                            outputIpr.set(x, y, (int) Math.round(output[0][c-1][x][y] * 255));
                        else if (bitDepth == 16)
                            outputIpr.setf(x, y, (int) Math.round(output[0][c-1][x][y] * 65535));
                        else if (bitDepth == 32)
                            outputIpr.setf(x, y, output[0][c-1][x][y]);
            }
        }

        return ImageFactory.createImage(outputImageName, outputIpl);

    }

    public static void addSliceToStack(Image imageStack, Image imageSlice, int slice, int frame) {
        ImagePlus stackIpl = imageStack.getImagePlus();
        ImagePlus sliceIpl = imageSlice.getImagePlus();

        for (int c = 0; c < sliceIpl.getNChannels(); c++) {
            stackIpl.setPosition(c + 1, slice + 1, frame + 1);
            sliceIpl.setPosition(c + 1, 1, 1);
            stackIpl.setProcessor(sliceIpl.getProcessor());
        }
    }

    @Override
    public Status process(Workspace workspace) {
        // Getting parameters
        String inputImageName = parameters.getValue(INPUT_IMAGE, workspace);
        String outputImageName = parameters.getValue(OUTPUT_IMAGE, workspace);
        String outputBitDepth = parameters.getValue(OUTPUT_BIT_DEPTH, workspace);
        boolean outputSubsetOfClasses = parameters.getValue(OUTPUT_SUBSET_OF_CLASSES, workspace);
        String outputClasses = parameters.getValue(OUTPUT_CLASSES, workspace);
        String modelPath = parameters.getValue(MODEL_PATH, workspace);
        int tileOverlap = parameters.getValue(TILE_OVERLAP, workspace);

        int bitDepth = Integer.parseInt(outputBitDepth);

        int[] classList = null;
        if (outputSubsetOfClasses) {
            outputClasses = outputClasses.trim();
            String[] strClassList = outputClasses.split(",");
            classList = new int[strClassList.length];
            for (int i = 0; i < strClassList.length; i++)
                classList[i] = Integer.parseInt(strClassList[i]);
        }

        // Getting input image
        Image inputImage = workspace.getImages().get(inputImageName);

        try {
            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            OrtSession session = environment.createSession(modelPath);

            // Tiling image if necessary
            Image tiledInputImage = tileImage(session, inputImage, tileOverlap);
            Image tiledOutputImage = createEmptyOutputImage(session, "Tiled output", tiledInputImage, bitDepth,
                    classList);

            int count = 0;
            int total = (int) (tiledInputImage.getNSlices() * tiledInputImage.getNFrames());
            for (int z = 0; z < tiledInputImage.getNSlices(); z++) {
                for (int t = 0; t < tiledInputImage.getNFrames(); t++) {
                    Image inputSlice = ExtractSubstack.extractSubstack(tiledInputImage, "TIled slice", "1-end",
                            String.valueOf(z + 1), String.valueOf(t + 1));

                    // Preparing input data
                    OnnxTensor inputTensor = getInputTensor(environment, session, inputSlice);

                    // Running model
                    HashMap<String, OnnxTensor> inputMap = new HashMap<>();
                    String inputName = session.getInputNames().iterator().next();
                    inputMap.put(inputName, inputTensor);
                    Result outputMap = session.run(inputMap);

                    // Preparing output data
                    Image outputSlice = getOutputImage(session, outputMap, outputImageName, bitDepth, classList);

                    // Putting slice into tiledOutputImage
                    addSliceToStack(tiledOutputImage, outputSlice, z, t);

                    writeProgressStatus(++count, total, "slices");

                }
            }

            Image outputImage = stitchImage(tiledOutputImage, outputImageName, tileOverlap, (int) inputImage.getWidth(),
                    (int) inputImage.getHeight());

            workspace.addImage(outputImage);

            // If the image is being saved as a new image, adding it to the workspace
            if (showOutput)
                outputImage.show();

            return Status.PASS;

        } catch (

        OrtException e) {
            MIA.log.writeError(e);
            return Status.FAIL;
        }

    }

    @Override
    protected void initialiseParameters() {
        parameters.add(new SeparatorP(INPUT_SEPARATOR, this));
        parameters.add(new InputImageP(INPUT_IMAGE, this));
        parameters.add(new MessageP(INPUT_NOTE, this,
                "Input images are expected to be 32-bit with values in the range 0-1.", ParameterState.MESSAGE));

        parameters.add(new SeparatorP(OUTPUT_SEPARATOR, this));
        parameters.add(new OutputImageP(OUTPUT_IMAGE, this));
        parameters.add(new ChoiceP(OUTPUT_BIT_DEPTH, this, OutputBitDepths.THIRTY_TWO, OutputBitDepths.ALL,
                "By default images will be saved as floating point 32-bit (probabilities in the range 0-1); however, they can be converted to 8-bit (probabilities in the range 0-255) or 16-bit (probabilities in the range 0-65535).  This is useful for saving memory or if the output probability map will be passed to image threshold module."));
        parameters.add(new BooleanP(OUTPUT_SUBSET_OF_CLASSES, this, false));
        parameters.add(new StringP(OUTPUT_CLASSES, this));

        parameters.add(new SeparatorP(MODEL_SEPARATOR, this));
        parameters.add(new FilePathP(MODEL_PATH, this));
        parameters.add(new IntegerP(TILE_OVERLAP, this, 64));

    }

    @Override
    public Parameters updateAndGetParameters() {
        Workspace workspace = null;

        Parameters returnedParameters = new Parameters();
        returnedParameters.add(parameters.getParameter(INPUT_SEPARATOR));
        returnedParameters.add(parameters.getParameter(INPUT_IMAGE));
        returnedParameters.add(parameters.getParameter(INPUT_NOTE));

        returnedParameters.add(parameters.getParameter(OUTPUT_SEPARATOR));
        returnedParameters.add(parameters.getParameter(OUTPUT_IMAGE));
        returnedParameters.add(parameters.getParameter(OUTPUT_BIT_DEPTH));
        returnedParameters.add(parameters.getParameter(OUTPUT_SUBSET_OF_CLASSES));
        if ((boolean) parameters.getValue(OUTPUT_SUBSET_OF_CLASSES, workspace))
            returnedParameters.add(parameters.getParameter(OUTPUT_CLASSES));

        returnedParameters.add(parameters.getParameter(MODEL_SEPARATOR));
        returnedParameters.add(parameters.getParameter(MODEL_PATH));
        returnedParameters.add(parameters.getParameter(TILE_OVERLAP));

        return returnedParameters;

    }

    @Override
    public ImageMeasurementRefs updateAndGetImageMeasurementRefs() {
        return null;
    }

    @Override
    public ObjMeasurementRefs updateAndGetObjectMeasurementRefs() {
        return null;
    }

    @Override
    public ObjMetadataRefs updateAndGetObjectMetadataRefs() {
        return null;
    }

    @Override
    public MetadataRefs updateAndGetMetadataReferences() {
        return null;
    }

    @Override
    public ParentChildRefs updateAndGetParentChildRefs() {
        return null;
    }

    @Override
    public PartnerRefs updateAndGetPartnerRefs() {
        return null;
    }

    @Override
    public boolean verify() {
        return true;
    }
}
