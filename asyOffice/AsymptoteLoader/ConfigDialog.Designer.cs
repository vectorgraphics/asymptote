namespace AsymptoteLoader
{
    partial class ConfigDialog
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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btnSpecifyRuntime = new System.Windows.Forms.CheckBox();
            this.btnSelectAsy = new System.Windows.Forms.Button();
            this.txtCustomAsy = new System.Windows.Forms.TextBox();
            this.btnAccept = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnAbout = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnSpecifyRuntime);
            this.groupBox1.Controls.Add(this.btnSelectAsy);
            this.groupBox1.Controls.Add(this.txtCustomAsy);
            this.groupBox1.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.groupBox1.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(351, 95);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Asymptote Runtime";
            // 
            // btnSpecifyRuntime
            // 
            this.btnSpecifyRuntime.AutoSize = true;
            this.btnSpecifyRuntime.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSpecifyRuntime.Location = new System.Drawing.Point(18, 22);
            this.btnSpecifyRuntime.Name = "btnSpecifyRuntime";
            this.btnSpecifyRuntime.Size = new System.Drawing.Size(157, 19);
            this.btnSpecifyRuntime.TabIndex = 2;
            this.btnSpecifyRuntime.Text = "Specify Custom Runtime";
            this.btnSpecifyRuntime.UseVisualStyleBackColor = true;
            this.btnSpecifyRuntime.CheckedChanged += new System.EventHandler(this.btnSpecifyRuntime_CheckedChanged);
            // 
            // btnSelectAsy
            // 
            this.btnSelectAsy.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnSelectAsy.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSelectAsy.Location = new System.Drawing.Point(312, 60);
            this.btnSelectAsy.Name = "btnSelectAsy";
            this.btnSelectAsy.Size = new System.Drawing.Size(33, 20);
            this.btnSelectAsy.TabIndex = 1;
            this.btnSelectAsy.Text = "...";
            this.btnSelectAsy.UseVisualStyleBackColor = true;
            this.btnSelectAsy.Click += new System.EventHandler(this.btnSelectAsy_Click);
            // 
            // txtCustomAsy
            // 
            this.txtCustomAsy.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCustomAsy.Location = new System.Drawing.Point(18, 60);
            this.txtCustomAsy.Name = "txtCustomAsy";
            this.txtCustomAsy.Size = new System.Drawing.Size(282, 23);
            this.txtCustomAsy.TabIndex = 0;
            // 
            // btnAccept
            // 
            this.btnAccept.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnAccept.Location = new System.Drawing.Point(282, 119);
            this.btnAccept.Name = "btnAccept";
            this.btnAccept.Size = new System.Drawing.Size(75, 23);
            this.btnAccept.TabIndex = 1;
            this.btnAccept.Text = "Accept";
            this.btnAccept.UseVisualStyleBackColor = true;
            this.btnAccept.Click += new System.EventHandler(this.btnAccept_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCancel.Location = new System.Drawing.Point(201, 119);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 2;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // btnAbout
            // 
            this.btnAbout.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnAbout.Location = new System.Drawing.Point(12, 119);
            this.btnAbout.Name = "btnAbout";
            this.btnAbout.Size = new System.Drawing.Size(75, 23);
            this.btnAbout.TabIndex = 3;
            this.btnAbout.Text = "About";
            this.btnAbout.UseVisualStyleBackColor = true;
            // 
            // ConfigDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(375, 154);
            this.Controls.Add(this.btnAbout);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnAccept);
            this.Controls.Add(this.groupBox1);
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ConfigDialog";
            this.Text = "ConfigDialog";
            this.Load += new System.EventHandler(this.ConfigDialog_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.CheckBox btnSpecifyRuntime;
        private System.Windows.Forms.Button btnSelectAsy;
        private System.Windows.Forms.TextBox txtCustomAsy;
        private System.Windows.Forms.Button btnAccept;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnAbout;
    }
}