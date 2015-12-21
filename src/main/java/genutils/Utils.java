/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package genutils;

import com.google.common.primitives.Ints;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;
import java.util.Map.Entry;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;


/**
 *
 * @author Thanos
 */
public class Utils {
    
    
    /**
     * Replicate row rowMult times
     * @param row
     * @param rowMult
     * @return 
     */
    public static DoubleMatrix repmatRow(DoubleMatrix row, int rowMult) {
        
        return row.repmat(rowMult, 1);
    }
    

    public static DoubleMatrix concatenateMatrices(DoubleMatrix a, DoubleMatrix b ) {
        
        int a_rows = a.getRows();
        int a_cols = a.getColumns();
             
        int b_rows = b.getRows();
        int b_cols = b.getColumns();
        
        if(a_cols != b_cols) {
            return null;
        }
        
        DoubleMatrix response = new DoubleMatrix(a_rows + b_rows, a_cols); 
        for(int r = 0; r < a_rows; r++) {
            for(int c = 0; c < a_cols; c++) {
                response.put(r, c, a.get(r, c));
            }
        }
                
        for(int r = 0; r < b_rows; r++) {
            for(int c = 0; c < b_cols; c++) {
                response.put(r + a_rows, c, b.get(r, c));
            }
        }
        return response;        
    }
    
    
    
    /** 
     * Sums up all the entries of the matrix
     * @param m
     * @return 
     */
    public static double sumRowsAndColumns(DoubleMatrix m ) {
        double sum = 0.0;
        
        for(int i = 0; i < m.getRows(); i++) {
            for(int j = 0; j < m.getColumns(); j++) {
                sum += m.get(i, j);
            }
        }
        
        return sum;
    }
    
    
    /**
     * It is a binary (0/1) matrix that will be used as a mask, for elementwise
     * multiplication. 
     * @param row is the rating values of a user [0 0 5 3 2 ... 0 3 5]
     * @return 
     */
    public static DoubleMatrix createRowMaskMatrix(DoubleMatrix row, int ratingSpread) {
        
        //DoubleMatrix mask = DoubleMatrix.zeros(5, row.getColumns());
        DoubleMatrix mask = DoubleMatrix.zeros(ratingSpread, row.getColumns());
        for(int col = 0; col < row.getColumns(); col++) {
            
            double value  = (int) row.get(0, col);
            if(value == 0.0 ) { //skip if the rating is missing
                continue;
            }
            
            int rating = (int) value;
            mask.put(rating - 1, col, 1.0);
        }
        
        return mask;
    }
    
    
    public static DoubleMatrix createMask(DoubleMatrix m, double mask) {
        
        DoubleMatrix pattern = DoubleMatrix.ones(m.getRows(), m.getColumns()).mul(mask);        
        return pattern.eq(m);
    }
    
    /** 
     * It returns a binary matrix, with 1's to non-zero entries of the original
     * matrix
     * @param m
     * @return 
     */
    public static DoubleMatrix binaryMe(DoubleMatrix m) {
        DoubleMatrix response = DoubleMatrix.zeros(m.getRows(), m.getColumns());                                
        for(int i = 0; i < m.getRows(); i++) {
            for(int j = 0; j < m.getColumns(); j++) {                
                if(m.get(i, j) > 0.0) {
                    response.put(i, j,1.0);
                }
            }
        }
        
        return response;
    }
    
       
    
    /**
     * Given a row, containing user ratings it produces a HashMap<Integer, List<Integer>>
     * where as <b>key</b> we use the rating, and as <b>value</b> the indexes 
     * of the columns that were assigned the rating
     * @param row
     * @return 
     */
    public static HashMap<Integer, List<Integer>> summarizeRatings(DoubleMatrix row) {
        
        HashMap<Integer, List<Integer>> response = new HashMap<Integer, List<Integer>>(5);
        for(int d = 0; d < row.getColumns(); d++) {
            
            double value = row.get(0, d);
            //if the entry is empty ignore..
            if(value == 0.0) {
                continue;
            }
            
            //cast value to integer
            int key = (int)value;
            List<Integer> get = response.get(key);
            if(get == null) {
                get = new ArrayList<Integer>();
            }
            
            get.add(d);
            response.put(key, get);
            
        }
        
        
        return response;
        
    }
    

    /**
     * The 'indicator' matrix is a binary matrix to indicate whether a user has
     * addressed a rating for a movie. The softMax function is computed for these 
     * entries only
     * @param data
     * @param indicator
     * @return 
     */
    public static HashMap<Integer, DoubleMatrix> softMax(HashMap<Integer, DoubleMatrix> data, DoubleMatrix indicator) {
        

        DoubleMatrix rat1 = data.get(1);
        DoubleMatrix rat2 = data.get(2);
        DoubleMatrix rat3 = data.get(3);
        DoubleMatrix rat4 = data.get(4);
        DoubleMatrix rat5 = data.get(5);
               
        int nrows = indicator.getRows();
        int ncols = indicator.getColumns();
        
        DoubleMatrix res1 = DoubleMatrix.zeros(nrows, ncols);
        DoubleMatrix res2 = DoubleMatrix.zeros(nrows, ncols);
        DoubleMatrix res3 = DoubleMatrix.zeros(nrows, ncols);
        DoubleMatrix res4 = DoubleMatrix.zeros(nrows, ncols);
        DoubleMatrix res5 = DoubleMatrix.zeros(nrows, ncols);
        

        
        for(int r = 0; r <nrows; r++ ) {
            for(int c = 0; c < ncols; c++) {
                if(indicator.get(r, c) == 0.0) {
                    continue;
                }
                
                double get1 = Math.exp(rat1.get(r, c));
                double get2 = Math.exp(rat2.get(r, c));
                double get3 = Math.exp(rat3.get(r, c));
                double get4 = Math.exp(rat4.get(r, c));
                double get5 = Math.exp(rat5.get(r, c));
                
                double sum = get1 + get2 + get3 + get4 + get5;
                res1.put(r,c, get1/sum);
                res2.put(r,c, get2/sum);
                res3.put(r,c, get3/sum);
                res4.put(r,c, get4/sum);
                res5.put(r,c, get5/sum);
            }
            
        }
        
                
        HashMap<Integer, DoubleMatrix> response = new HashMap<Integer, DoubleMatrix>(5);
        response.put(1, res1);        
        response.put(2, res2);      
        response.put(3, res3);        
        response.put(4, res4);           
        response.put(5, res5);   
        
        return response;
        
    }
    
    /**
     * Given a data matrix, it returns the softmax for each column vector
     * @param x
     * @return 
     */
    public static DoubleMatrix softmax(DoubleMatrix x ) {
        
        int rows = x.getRows();
        int columns = x.getColumns();
        
        DoubleMatrix response = DoubleMatrix.zeros(rows, columns);
        for(int c = 0; c < x.getColumns(); c++ ) {
            
            //current column
            DoubleMatrix column = x.getColumn(c);
            double sum = 0.0;            
           
            
            // if all the entries in the column are zero, ignore them..
            if(column.columnSums().get(0,0) == 0.0) {
                           
                for(int row = 0; row < rows; row++ ) {
                    response.put(row, c, 0.0);
                }
                
                continue;
            }
            
            for(int row = 0; row < rows; row++ ) {
                sum += Math.exp(column.get(row, 0));
            }
                                   
            for(int row = 0; row < rows; row++ ) {
                response.put(row, c, Math.exp(column.get(row, 0))/sum);
            }
            
        }
        return response;
    }
    
    
    /**
     * Returns a row matrix, with size 'size', with zero entries, except those
     * specified in the 'indices' list
     * @param size
     * @param indices
     * @return 
     */
    public static DoubleMatrix createRowMatrix(int size, List<Integer> indices) {
        
        DoubleMatrix row = DoubleMatrix.zeros(1, size);
        for(int col : indices) {
            row.put(0, col, 1.0);
        }        
        return row;
    }
    
    public static List<Integer> unique(org.jblas.DoubleMatrix x) {
                
        HashSet<Double> set = new HashSet<Double>();
        
        List<Double> elementsAsList = x.elementsAsList();
        set.addAll(elementsAsList);
        
        List<Integer> list = new ArrayList<Integer>();
        for(double d : set) {
            list.add((int)d);
        }
        Collections.sort(list);
        
        return list;
    }
        
 
 
    public static String rowMatrixToCsv(DoubleMatrix m) { 
        
        String res = "";                    
        for(int j = 0; j < m.getColumns(); j++) {
                res += m.get(0, j);
                if(j <  (m.getColumns() - 1)) {
                    res += ",";
                }                        
        }
        
        return res;
    }
     
    
   
    public static List<Integer> getSequence(int index1, int index2) {
        
        List<Integer> response = new ArrayList<Integer>();
        if(index1 < index2) {
            for(int item = index1; item <= index2; item++ )
                response.add(item);
        }else {                      
            for(int item = index1; item >= index2; item-- )
                response.add(item);
        }
        
        return response;        
    }
    
    
    public static int[] randomPermute(int[] sequence) {
               
        List<Integer> asList = new ArrayList<Integer>();
        for(int i = 0; i < sequence.length; i++) {
            asList.add(sequence[i]);
        }
                        
        //do the shuffling
        Collections.shuffle(asList);
        sequence = Ints.toArray(asList); 
        
        return sequence;
    }
    
    
        
    public static org.jblas.DoubleMatrix logistic(org.jblas.DoubleMatrix x) {

        org.jblas.DoubleMatrix res = org.jblas.DoubleMatrix.zeros(x.getRows(), x.getColumns());
       
        for(int i = 0; i < x.getRows(); i++) {
            for(int j = 0; j < x.getColumns(); j++) {                
                res.put(i, j, logistic(x.get(i, j)));
            }
        }
        return res;
    }
    
        
    public static double logistic(double x) {

        return 1/(1+Math.exp(-x));

    }
    
    
       
    public static org.jblas.DoubleMatrix softmaxPmtk(org.jblas.DoubleMatrix M) {
               
        org.jblas.DoubleMatrix exp = org.jblas.MatrixFunctions.exp(M);                
        org.jblas.DoubleMatrix sums = exp.rowSums();
                
        return exp.diviColumnVector(sums);
    
    }    
        
    
    
    public static org.jblas.DoubleMatrix softmax_sample(org.jblas.DoubleMatrix probmat) {
        
        org.jblas.DoubleMatrix oneofn = org.jblas.DoubleMatrix.zeros(probmat.getRows(), probmat.getColumns());             
        org.jblas.DoubleMatrix repmat = probmat.rowSums().repmat(1, probmat.getColumns());
        probmat = probmat.divi(repmat);
        
        org.jblas.DoubleMatrix rand = org.jblas.DoubleMatrix.rand(probmat.getRows(), 1);
        
        for(int i = 0; i < probmat.getRows(); i++) {
                      
            org.jblas.DoubleMatrix probs = probmat.getRow(i);
            DoubleMatrix sample = probs.cumulativeSum();
            sample = sample.ge(rand.get(i));
            
            int index = 0;                     
            while(sample.get(index) < 1.0) {
                index++;
                
                if(index == sample.getColumns() - 1) {
                    break;
                }
            }
            
            oneofn.put(i, index, 1);
        }                                
        
        return oneofn;
        
    }    
 
}
