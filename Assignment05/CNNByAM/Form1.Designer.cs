namespace CNNByAM
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.btnTrainCNN = new System.Windows.Forms.Button();
            this.btnTestDigitAfterTraining = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.picTest = new System.Windows.Forms.PictureBox();
            this.btnSaveNetworkAfterTraining = new System.Windows.Forms.Button();
            this.btnLoadTrainedNetwork = new System.Windows.Forms.Button();
            this.btnTest = new System.Windows.Forms.Button();
            this.btnRegularNN = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.picTest)).BeginInit();
            this.SuspendLayout();
            // 
            // btnTrainCNN
            // 
            this.btnTrainCNN.BackColor = System.Drawing.Color.DodgerBlue;
            this.btnTrainCNN.ForeColor = System.Drawing.Color.White;
            this.btnTrainCNN.Location = new System.Drawing.Point(140, 41);
            this.btnTrainCNN.Name = "btnTrainCNN";
            this.btnTrainCNN.Size = new System.Drawing.Size(433, 59);
            this.btnTrainCNN.TabIndex = 0;
            this.btnTrainCNN.Text = "Train CNN";
            this.btnTrainCNN.UseVisualStyleBackColor = false;
            this.btnTrainCNN.Click += new System.EventHandler(this.btnTrainCNN_Click);
            // 
            // btnTestDigitAfterTraining
            // 
            this.btnTestDigitAfterTraining.BackColor = System.Drawing.Color.Green;
            this.btnTestDigitAfterTraining.ForeColor = System.Drawing.Color.White;
            this.btnTestDigitAfterTraining.Location = new System.Drawing.Point(140, 443);
            this.btnTestDigitAfterTraining.Name = "btnTestDigitAfterTraining";
            this.btnTestDigitAfterTraining.Size = new System.Drawing.Size(422, 55);
            this.btnTestDigitAfterTraining.TabIndex = 10;
            this.btnTestDigitAfterTraining.Text = "TestDigit After Training";
            this.btnTestDigitAfterTraining.UseVisualStyleBackColor = false;
            this.btnTestDigitAfterTraining.Click += new System.EventHandler(this.btnTestDigitAfterTraining_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(592, 349);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(129, 32);
            this.label1.TabIndex = 9;
            this.label1.Text = "unknown";
            // 
            // picTest
            // 
            this.picTest.Location = new System.Drawing.Point(598, 398);
            this.picTest.Name = "picTest";
            this.picTest.Size = new System.Drawing.Size(100, 100);
            this.picTest.TabIndex = 8;
            this.picTest.TabStop = false;
            // 
            // btnSaveNetworkAfterTraining
            // 
            this.btnSaveNetworkAfterTraining.BackColor = System.Drawing.Color.DodgerBlue;
            this.btnSaveNetworkAfterTraining.ForeColor = System.Drawing.Color.White;
            this.btnSaveNetworkAfterTraining.Location = new System.Drawing.Point(140, 136);
            this.btnSaveNetworkAfterTraining.Name = "btnSaveNetworkAfterTraining";
            this.btnSaveNetworkAfterTraining.Size = new System.Drawing.Size(433, 55);
            this.btnSaveNetworkAfterTraining.TabIndex = 11;
            this.btnSaveNetworkAfterTraining.Text = "Save Network (after training)";
            this.btnSaveNetworkAfterTraining.UseVisualStyleBackColor = false;
            this.btnSaveNetworkAfterTraining.Click += new System.EventHandler(this.btnSaveNetworkAfterTraining_Click);
            // 
            // btnLoadTrainedNetwork
            // 
            this.btnLoadTrainedNetwork.BackColor = System.Drawing.Color.Green;
            this.btnLoadTrainedNetwork.ForeColor = System.Drawing.Color.White;
            this.btnLoadTrainedNetwork.Location = new System.Drawing.Point(140, 293);
            this.btnLoadTrainedNetwork.Name = "btnLoadTrainedNetwork";
            this.btnLoadTrainedNetwork.Size = new System.Drawing.Size(433, 55);
            this.btnLoadTrainedNetwork.TabIndex = 12;
            this.btnLoadTrainedNetwork.Text = "Load Trained Network ";
            this.btnLoadTrainedNetwork.UseVisualStyleBackColor = false;
            this.btnLoadTrainedNetwork.Click += new System.EventHandler(this.btnLoadTrainedNetwork_Click);
            // 
            // btnTest
            // 
            this.btnTest.Location = new System.Drawing.Point(680, 59);
            this.btnTest.Name = "btnTest";
            this.btnTest.Size = new System.Drawing.Size(183, 48);
            this.btnTest.TabIndex = 13;
            this.btnTest.Text = "Test";
            this.btnTest.UseVisualStyleBackColor = true;
            this.btnTest.Click += new System.EventHandler(this.btnTest_Click);
            // 
            // btnRegularNN
            // 
            this.btnRegularNN.BackColor = System.Drawing.Color.Yellow;
            this.btnRegularNN.Location = new System.Drawing.Point(612, 136);
            this.btnRegularNN.Name = "btnRegularNN";
            this.btnRegularNN.Size = new System.Drawing.Size(251, 55);
            this.btnRegularNN.TabIndex = 14;
            this.btnRegularNN.Text = "Train Regular NN";
            this.btnRegularNN.UseVisualStyleBackColor = false;
            this.btnRegularNN.Click += new System.EventHandler(this.btnRegularNN_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(16F, 31F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(906, 573);
            this.Controls.Add(this.btnRegularNN);
            this.Controls.Add(this.btnTest);
            this.Controls.Add(this.btnLoadTrainedNetwork);
            this.Controls.Add(this.btnSaveNetworkAfterTraining);
            this.Controls.Add(this.btnTestDigitAfterTraining);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.picTest);
            this.Controls.Add(this.btnTrainCNN);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.picTest)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnTrainCNN;
        private System.Windows.Forms.Button btnTestDigitAfterTraining;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.PictureBox picTest;
        private System.Windows.Forms.Button btnSaveNetworkAfterTraining;
        private System.Windows.Forms.Button btnLoadTrainedNetwork;
        private System.Windows.Forms.Button btnTest;
        private System.Windows.Forms.Button btnRegularNN;
    }
}

