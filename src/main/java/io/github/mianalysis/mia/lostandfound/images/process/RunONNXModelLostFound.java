package io.github.mianalysis.mia.lostandfound.images.process;

import java.util.HashMap;

import org.scijava.Priority;
import org.scijava.plugin.Plugin;

import io.github.mianalysis.mia.module.images.process.RunONNXModel;
import io.github.mianalysis.mia.module.lostandfound.LostAndFoundItem;

@Plugin(type = LostAndFoundItem.class, priority = Priority.LOW, visible = true)
public class RunONNXModelLostFound extends LostAndFoundItem {

    @Override
    public String getModuleName() {
        return new RunONNXModel(null).getClass().getSimpleName();
    }

    @Override
    public String[] getPreviousModuleNames() {
        return new String[]{""};
    }

    @Override
    public HashMap<String, String> getPreviousParameterNames() {
        HashMap<String,String> parameterNames = new HashMap<String,String>();
        parameterNames.put("Input node", null);
        
        return parameterNames;

    }

    @Override
    public HashMap<String, HashMap<String, String>> getPreviousParameterValues() {
        return new HashMap<String, HashMap<String, String>>();
    }
    
}
