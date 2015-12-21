/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package deeplearning.tools.rbmforcollaborativefiltering;

import genutils.BigFile;
import genutils.Rating;
import genutils.RbmOptions;
import genutils.Utils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;

/**
 *
 * @author thanos
 */
public class CollaborativeFilteringRBM {
    
   
    static final java.util.logging.Logger _logger = Logger.getLogger(CollaborativeFilteringRBM.class.getName());      
    
    // constants related to the training procedure
    final double epsilonw = 0.1;   // learning rate for weights
    final double epsilonvb = 0.1;  // learning rate for biases for visible units
    final double epsilonhb = 0.1;  // learning rate for biases for hidden units
    //double weightcost = 0.0002;
    final double weightcost = 0.002;
    final double initialmomentum = 0.5;
    final double finalmomentum = 0.9;
    final double modifier = 20.0;
    
    // rbm configuration details
    int numhid; // number of units in the hidden layer
    int numdims; // number of visible units
    
    // biases & connection weights
    HashMap<Integer, DoubleMatrix> Wijk;
    DoubleMatrix hidbiases;
    HashMap<Integer, DoubleMatrix> visbiases;
    
    DoubleMatrix matrix;    
    HashMap<String, Integer> feature2Index;
    HashMap<String, Integer> user2Index;    
    HashMap<Integer, String> index2User;
    
    /**
     * Fits a separate RBM for each user, with 'tied' weights and biases
     * for the hidden and visible units. 
     * @param matrix
     * @param rbmOptions
     * @return 
     */
    public void fit(RbmOptions rbmOptions) throws IOException {
                

        int startAveraging = rbmOptions.maxepoch - rbmOptions.avglast;             

        int N = matrix.getRows(); //    number of users
        int d = matrix.getColumns();//  number of dimensions
        
        _logger.info("got ratings from " + N + " users for " + d + " movies..");
        //Create batches
        //Batches batches = Utils.createBatches(N, rbmOptions.batchsize);      
        this.numdims = d;
        this.numhid = rbmOptions.numhid;
 
        //initialize visible-hidden symmetric weights
        
        //1a. visual to hidden connection weights (1 per rating)
        this.Wijk = new HashMap<Integer, DoubleMatrix>(5);        
        for(int rating = 1; rating <=5; rating++) {
            
            Wijk.put(rating, DoubleMatrix.randn(numdims, numhid).mul(0.1));
        }
        
        
        HashMap<Integer, DoubleMatrix> Wijk_inc = new HashMap<Integer, DoubleMatrix>(5);        
        for(int rating = 1; rating <=5; rating++) {            
            Wijk_inc.put(rating, DoubleMatrix.zeros(numdims, numhid));
        }
        
                
        //1b. gradients for visual to hidden connection weights (1 per rating)
        HashMap<Integer, DoubleMatrix> posprods = new HashMap<Integer, DoubleMatrix>(5);        
        for(int rating = 1; rating <=5; rating++) {
            
            posprods.put(rating, DoubleMatrix.zeros(numdims, numhid));
        }
        
        HashMap<Integer, DoubleMatrix> negprods = new HashMap<Integer, DoubleMatrix>(5);     
        for(int rating = 1; rating <=5; rating++) {
            
            negprods.put(rating, DoubleMatrix.zeros(numdims, numhid));
        }
        
        //2. biases for the hidden units
        this.hidbiases = DoubleMatrix.zeros(1, numhid);
        DoubleMatrix hidbiasinc = DoubleMatrix.zeros(1, numhid);        
                        
        //3. biases for the visible units (1 per rating)         
        this.visbiases = new HashMap<Integer, DoubleMatrix>(5);        
        HashMap<Integer, DoubleMatrix> visbiasesInc = new HashMap<Integer, DoubleMatrix>(5);       
        for(int rating = 1; rating <=5; rating++) {
            
            visbiases.put(rating, DoubleMatrix.zeros(1, numdims));
            visbiasesInc.put(rating, DoubleMatrix.zeros(1, numdims));
        }        
        
                
        //train for 'maxepoch' epochs
        for (int epoch = 1; epoch <= rbmOptions.maxepoch; epoch++) {

            _logger.info("Starting epoch " + (epoch + 1) + "\n");
            double errsum = 0;
            
            // randomize the visiting order and then treat
            // each training case separately..
            List<Integer> visitingSeq = Utils.getSequence(0, N - 1);
            Collections.shuffle(visitingSeq);
            
            for (int r = 0; r < N; r++) {
                                
                
                // each 'row' is in the form [0 0 5 4 2 0 0 3 ... 0 0 1 2 5],
                // If the value is > 0, it is the rating for that movie (column)
                // otherwise the rating is missing
                DoubleMatrix row = matrix.getRow(visitingSeq.get(r));
                
                if(rbmOptions.debug)
                    _logger.info("Examining row.." + row.toString());
                
                
                // a row matrix (1 x numdims) with 1's in the non-zero columns
                DoubleMatrix indicator = Utils.binaryMe(row);  
                if(rbmOptions.debug)
                    _logger.info("Indicator..." + indicator.toString());
                
                
                DoubleMatrix V = Utils.createRowMaskMatrix(row, 5);
                
                //positive phase
                DoubleMatrix poshidprobs = DoubleMatrix.zeros(1, numhid);
                
                //add the biases
                poshidprobs.addi(hidbiases);                
                
                // the key is the 'rating' and and the value is the list of 
                // columns (movies) with this rating
                HashMap<Integer, List<Integer>> summarizeRatings = Utils.summarizeRatings(row);

                for (int rating = 1; rating <= 5; rating++) {

                    // 'columnList' contains all the column indices with
                    //rating equal to the current 'rating'
                    List<Integer> columnList = summarizeRatings.get(rating);
                    if (columnList == null) {
                        continue;
                    }

                    // 'rowMatrix' will have one's in the columns where the rating
                    // is equal to the current 'rating', and zero otherwise
                    DoubleMatrix rowMatrix = Utils.createRowMatrix(numdims, columnList);
                    DoubleMatrix wij = Wijk.get(rating);
                    
                    // get the contribution from the active visible units
                    DoubleMatrix product = rowMatrix.mmul(wij);
                    poshidprobs.addi(product);
                }

                
                //take the logistic, to form probabilities for the hidden units
                poshidprobs = Utils.logistic(poshidprobs);
                
                
                // posprods = data' * poshidprobs
                for(int rating = 1; rating <=5; rating++) {  
                    DoubleMatrix posprod = posprods.get(rating);
                    DoubleMatrix vRow = V.getRow(rating - 1);                    
                    posprod.addi(vRow.transpose().mmul(poshidprobs));
                    posprods.put(rating, posprod);
                }
                
                
                if(rbmOptions.debug)
                    _logger.info("poshidprobs..." + poshidprobs.toString());
                
                //end of positive phase
                DoubleMatrix poshidstates = poshidprobs.ge(DoubleMatrix.rand(1, numhid));
                if(rbmOptions.debug)
                    _logger.info("poshidstates..." + poshidstates.toString());                                                                
                
                    
                if(rbmOptions.debug)                
                    _logger.info("*** END OF POSITIVE PHASE \n\n\n");    
                
                
                
                //start negative phase        
                DoubleMatrix negdata = DoubleMatrix.zeros(5, numdims);
                
                for(int index = 0; index < indicator.getColumns(); index++ ) {
                    
                    // do not reconstruct missing ratings
                    if(indicator.get(0, index) == 0.0) { 
                        continue;
                    }
                    
                    
                    for(int rat = 0; rat < 5; rat++) {
                    
                        int rating = rat + 1;
                        
                        //get the bias for the specfic visible unit/rating
                        DoubleMatrix vbias = visbiases.get(rating);                                                
                        double bias = vbias.get(0, index);
                        
                        DoubleMatrix wij = Wijk.get(rating);
                        
                        double sum = bias;
                        for(int hid = 0; hid < poshidstates.getColumns(); hid++) {
                            
                            //if the hidden is turned on, use it
                            if(poshidstates.get(0, hid) > 0.0) {
                             
                                sum += wij.get(index, hid);
                            }
                        }
                        
                        negdata.put(rat, index, sum);
                    }                                    
                }
                
                
                // zero negata values for the zero ratings
                negdata = Utils.softmax(negdata);
                
                
                DoubleMatrix neghidprobs = DoubleMatrix.zeros(1, numhid);
                
                // add the biases for the hidden units
                neghidprobs.addi(hidbiases);
                
                for(int index = 0; index < negdata.getColumns(); index++ ) {                    
                    
                    // if the rating is missing ignore it
                    if(negdata.getColumn(index).columnSums().get(0,0) == 0.0) {
                        continue;
                    }
                    
                    for (int k = 0; k < 5; k++) {
                    
                        int rating = k + 1;
                        DoubleMatrix wij = Wijk.get(rating);
                        double visible_prob = negdata.get(k, index);
                        if(rbmOptions.debug)
                            _logger.info("visible prob = " + visible_prob);
                        DoubleMatrix wToHidden = wij.getRow(index);
                        if(rbmOptions.debug)
                            _logger.info("wToHidden is \n " + wToHidden.toString());
                        DoubleMatrix contributionToHiddenUnits = wToHidden.mul(visible_prob);   
                        if(rbmOptions.debug)
                            _logger.info("Adding.........." + contributionToHiddenUnits.toString());                        
                                                
                        neghidprobs.addi(contributionToHiddenUnits);
                     }
                }
                              
                neghidprobs = Utils.logistic(neghidprobs);
                if(rbmOptions.debug) {
                    _logger.info("neghidprobs.. " + neghidprobs.toString());
                }

                // negprods = negdata' * neghidprobs
                for(int rating = 1; rating <=5; rating++) {  
                    DoubleMatrix negprod = negprods.get(rating);
                    DoubleMatrix vRow = negdata.getRow(rating - 1);                    
                    negprod.addi(vRow.transpose().mmul(neghidprobs));
                    negprods.put(rating, negprod);
                }
                
                
                //the end for each user                
                DoubleMatrix error = V.sub(negdata);                 
                double err = error.norm2();
                
                //debug
                if(Double.isNaN(err)) {
                    _logger.info("Examining row.." + row.toString());
                    _logger.info("poshidprobs..." + poshidprobs.toString());
                    _logger.info("poshidstates..." + poshidstates.toString());                        
                    _logger.info("neghidprobs.. " + neghidprobs.toString());
                    _logger.info("error.. " + error.toString());
                } 
                else {
                    errsum += err;
                }
                                
                            
                //set momentum
                double momentum = 0.0;
                if (epoch > startAveraging) {
                    momentum = finalmomentum;
                } else {
                    momentum = initialmomentum;
                }
            
                
                //calculate gradients
                hidbiasinc = (hidbiasinc.mul(momentum)).add((poshidprobs.sub(neghidprobs)).mul(epsilonhb*modifier/N));
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix inc = visbiasesInc.get(rating);
                    DoubleMatrix temp1 = inc.mul(momentum);
                    DoubleMatrix temp2 = (V.getRow(rating - 1).sub(negdata.getRow(rating - 1))).mul(epsilonvb*modifier/N);
                    inc = temp1.add(temp2);
                    visbiasesInc.put(rating, inc);
                }
                
                             
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix inc = Wijk_inc.get(rating);
                    DoubleMatrix temp1 = inc.mul(momentum);
                    DoubleMatrix temp2 = (posprods.get(rating).sub(negprods.get(rating))).mul(epsilonw*modifier/N);
                    DoubleMatrix temp3 = Wijk.get(rating).mul(weightcost);
                    inc = temp1.add(temp2).sub(temp3);
                    Wijk_inc.put(rating, inc);
                }
                                  
                
                                
                //update connection weights
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix wijk = Wijk.get(rating);
                    Wijk.put(rating, wijk.add(Wijk_inc.get(rating)));
                }
                
                hidbiases = hidbiases.add(hidbiasinc);
                for(int rating = 1; rating <=5; rating++ ) {
                    DoubleMatrix vis = visbiases.get(rating);
                    visbiases.put(rating, vis.add(visbiasesInc.get(rating)));
                }
                    
            }
            
                      
            // reset 'Winj_inc' matrix 
            for(int rating = 1; rating <=5; rating++ ) {
                    Wijk_inc.put(rating, DoubleMatrix.zeros(numdims, numhid)); 
            }
            
            _logger.info("Epoch " + epoch + " error " + errsum + "\n");
            
        } // end of epoch
                          
    }
                       
    
    /**
     * Reads the file with the user-item ratings. The expected format is 
     * 'user'<tab/>'item'<tab/>'rating
     * @param file 
     */
    public void loadRatings(String file) {
               
        HashMap<String, List<Rating>> ratingsMap = new HashMap<>(10000);
       
        try {
            BigFile f = new BigFile(file);
            HashSet<String> items = new HashSet<String>(5000);   
            
            Iterator<String> iterator = f.iterator();            
            while (iterator.hasNext()) {
                String line = iterator.next();
                String[] splits = line.split("\t");

                String userId = splits[0];
                String itemId = splits[1];
                double rating = Double.parseDouble(splits[2]);

                List<Rating> ratingsList = ratingsMap.get(userId);
                if(ratingsList == null) {
                    ratingsList = new ArrayList<>();
                }
                
                ratingsList.add(new Rating(itemId, rating));
                ratingsMap.put(userId, ratingsList);
                items.add(itemId);
            }

            System.out.println("Found " + ratingsMap.keySet().size() + " users " +
                    " and " + items.size() + " items..");
            
            // initialize all with zeros
            this.matrix = DoubleMatrix.zeros(ratingsMap.keySet().size(), items.size());
            
            user2Index = new HashMap<>(ratingsMap.keySet().size());
            index2User = new HashMap<>(ratingsMap.keySet().size());
            
            
            int rowIndex = 0;
            for(Map.Entry<String, List<Rating>> entry : ratingsMap.entrySet()) {
                
                String userId = entry.getKey();
                user2Index.put(userId, rowIndex);
                index2User.put(rowIndex, userId);
                rowIndex++;
            }
            int usersNr = rowIndex;
            
            
            int columnIndex = 0;
            feature2Index = new HashMap<>(items.size());
            for(String item : items) {
                
                feature2Index.put(item, columnIndex);
                columnIndex++;
            }
                        
            for(int row = 0; row < usersNr; row++) {
                
                String userId = index2User.get(row);
                List<Rating> ratings = ratingsMap.get(userId);
                for(Rating rating : ratings) {
                    
                    String item = rating.itemId;
                    double val = rating.rating;
                    
                    matrix.put(row, feature2Index.get(item), val);                    
                }                
            }
                  
                         
        } catch (Exception ex) {

            ex.printStackTrace();
        }
        
    }

    public double predict(String userId, String itemId, PredictionType predictionType) {

        int userIndex = user2Index.get(userId);
        int itemIndex = feature2Index.get(itemId);

        DoubleMatrix user_ratings_so_far = this.matrix.getRow(userIndex);
                
        //positive phase
        DoubleMatrix poshidprobs = DoubleMatrix.zeros(1, numhid);

        //add the biases
        poshidprobs.addi(hidbiases);
                   
        // the key is the 'rating' and and the value is the list of 
        // indices that contain that rating
        HashMap<Integer, List<Integer>> summarize = Utils.summarizeRatings(user_ratings_so_far);

        for (int rating = 1; rating <= 5; rating++) {

            // 'columnList' contains all the column indices with
            //rating equal to the current 'rating'
            List<Integer> columnList = summarize.get(rating);
            if (columnList == null) {
                continue;
            }

            // 'rowMatrix' will have one's in the columns where the rating
            // is equal to the current 'rating', and zero otherwise
            DoubleMatrix rowMatrix = Utils.createRowMatrix(numdims, columnList);
            DoubleMatrix wij = Wijk.get(rating);

            //get the contribution from the active visible units
            DoubleMatrix product = rowMatrix.mmul(wij);
            poshidprobs.addi(product);
        }

        //take the logistic, to form probabilities for the hidden units
        poshidprobs = Utils.logistic(poshidprobs);
        DoubleMatrix poshidstates = poshidprobs.ge(DoubleMatrix.rand(1, numhid));
        
        DoubleMatrix negdata = DoubleMatrix.zeros(5, 1);

        for (int rat = 0; rat < 5; rat++) {

            int rating = rat + 1;

            //get the bias for the specfic visible unit/rating
            DoubleMatrix vbias = visbiases.get(rating);
            double bias = vbias.get(0, itemIndex);

            DoubleMatrix wij = Wijk.get(rating);

            double sum = bias;
            for (int hid = 0; hid < poshidstates.getColumns(); hid++) {

                //if the hidden is turned on, use it
                if (poshidstates.get(0, hid) > 0.0) {

                    sum += wij.get(itemIndex, hid);
                }
            }

            negdata.put(rat, 0, sum);
        }
        
        negdata = Utils.softmax(negdata);
        
        if(predictionType.equals(PredictionType.MAX)) {
            
            int max_index = 0;
            double max_value = negdata.get(0,0);
            
            for(int i = 1; i < negdata.getRows(); i++ ) {
                double current = negdata.get(i,0);
                if(current > max_value) {
                    max_index = i;
                    max_value = current;
                }
            }
            
            return (max_index + 1)*1.0;
                       
        }else if(predictionType.equals(PredictionType.MEAN)) {
            
            double mean = 0.0;
                       
            for(int i = 0; i < negdata.getRows(); i++ ) {

                mean += negdata.get(i,0) * (i + 1);
                
            }
            
            return mean;
            
            
        }
        
        return 0.0;
    }
                
                   
}
