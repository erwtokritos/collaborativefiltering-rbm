/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package drivers;

import deeplearning.tools.rbmforcollaborativefiltering.CollaborativeFilteringRBM;
import deeplearning.tools.rbmforcollaborativefiltering.PredictionType;
import genutils.RbmOptions;
import java.io.IOException;
import java.util.logging.Logger;

/**
 *
 * @author thanos
 */
public class TestCollaborativeFilteringRBM {
    
    private static final Logger _logger = Logger.getLogger(TestCollaborativeFilteringRBM.class.getName());    
    
    public static void main(String[] args) throws IOException {
        
        
        _logger.info("Loading data..");        
               
        RbmOptions options = new RbmOptions();
        options.maxepoch = 10;
        options.avglast = 5;
        options.numhid = 100;
        options.debug = false;
        
        //CollaborativeFilteringLayer fit = CollaborativeFilteringRBM.fit(data, options);
        CollaborativeFilteringRBM rbmCF = new CollaborativeFilteringRBM();
        rbmCF.loadRatings("./data/" + "u.data");
        rbmCF.fit(options);
        
        System.out.println("Max prediction = " + rbmCF.predict("166", "346", PredictionType.MAX));               
        System.out.println("Mean prediction = " + rbmCF.predict("166", "346", PredictionType.MEAN));               
       
    }
    
}
