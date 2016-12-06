//Program			: Neural Net (ADALINE)
//Name				: RUPANTA RWITEEJ DUTTA
//Date of Creation	: 10.02.2016

package adaline;

import Jama.Matrix;

public class learning {
	public static void main(String[] args) {
		int qRow = 8;
		int qCol = 3;
		double[][] x1 = {{1,1,1},{1,2,0},{2,-1,-1},{2,0,0},{-1,2,-1},{-2,1,1},{-1,-1,-1},{-2,-2,0}};
		
		Matrix x = new Matrix(qRow,qCol);
		for(int i=0;i<qRow;i++)
			for(int j=0;j<qCol;j++)
				x.set(i, j, x1[i][j]);
		
		double[][] t1 = {{-1,-1},{-1,-1},{-1,1},{-1,1},{1,-1},{1,-1},{1,1},{1,1}};
		int tRow = 8;
		int tCol = 2;
		Matrix t = new Matrix(tRow,tCol);
		for(int i=0;i<tRow;i++)
			for(int j=0;j<tCol;j++)
				t.set(i, j, t1[i][j]);
		
		double[][] w1 = {{0,0,0},{0,0,0}};
		int wRow = 2;
		int wCol = 3;
		Matrix w = new Matrix(wRow,wCol);
		for(int i=0;i<wRow;i++)
			for(int j=0;j<wCol;j++)
				w.set(i, j, w1[i][j]);
		
		double[][] b1 = {{0,0},{0,0}};
		int bRow = 1;
		int bCol = 2;
		Matrix b = new Matrix(bRow,bCol);
		for(int i=0;i<bRow;i++)
			for(int j=0;j<bCol;j++)
				b.set(i, j, b1[i][j]);
		
		double alpha = 1;
		
		int steps = 200, i=1;
		Matrix wNew, wOld,bNew,bOld;
		wNew = w;
		bNew = b;
		while(i<=steps){
			//alpha = 1/i;
			//alpha = 1;		
			//alpha =alpha - alpha/6;
			alpha = alpha*0.2;					
			wOld = wNew;
			bOld = bNew;
			Matrix X = x.getMatrix(i%8,i%8,0,2);
			Matrix Yin = b.plus(X.times(w.transpose()));
			Matrix T = t.getMatrix(i%8, i%8, 0, 1);
			Matrix sub = Yin.minus(T);
			Matrix Xt = X.transpose();
			Matrix temp = Xt.times(sub);
			Matrix temp2 = temp.times(2*alpha);
			wNew = wOld.minus(temp2.transpose());
			bNew = bOld.minus(sub.times(2*alpha));
			System.out.println("================");
			System.out.println("    EPOCH "+i);
			System.out.println("Weight:\n-------");
			for(int row=0;row<wNew.getRowDimension();row++){
				for(int col=0;col<wNew.getColumnDimension();col++){
					System.out.print(wNew.get(row,col)+"  ");
				}
				System.out.println();
			}
			System.out.println("Bias:\n-----\n["+bNew.get(0,0)+" "+bNew.get(0,1)+"]");
			System.out.println("================");
			i++;
		}
		Matrix delta = new Matrix(2,1);
		for(i=0;i<qRow;i++){
			
			Matrix X = x.getMatrix(i,i,0,2);
			Matrix temp = wNew.times(X.transpose());
			Matrix T = t.getMatrix(i, i, 0, 1);
			temp = temp.minus(T.transpose());
			temp.set(0, 0, Math.pow(temp.get(0, 0),2));
			temp.set(1, 0, Math.pow(temp.get(1, 0), 2));
			delta = delta.plus(temp);
		}
		System.out.println("================");
		System.out.println("    DELTA    \n["+delta.get(0, 0)/8+" "+delta.get(1, 0)/8+"]");
		System.out.println("================");
	}
}
