namespace AsyPPTLoader
{
    partial class AsyLoaderRibbon : Microsoft.Office.Tools.Ribbon.RibbonBase
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        public AsyLoaderRibbon()
            : base(Globals.Factory.GetRibbonFactory())
        {
            InitializeComponent();
        }

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

        #region Component Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.tab1 = this.Factory.CreateRibbonTab();
            this.AsymptoteGroup = this.Factory.CreateRibbonGroup();
            this.btnAddAsy = this.Factory.CreateRibbonButton();
            this.btnLoadAsyCode = this.Factory.CreateRibbonButton();
            this.btnConfigure = this.Factory.CreateRibbonButton();
            this.tab1.SuspendLayout();
            this.AsymptoteGroup.SuspendLayout();
            this.SuspendLayout();
            // 
            // tab1
            // 
            this.tab1.ControlId.ControlIdType = Microsoft.Office.Tools.Ribbon.RibbonControlIdType.Office;
            this.tab1.Groups.Add(this.AsymptoteGroup);
            this.tab1.Label = "TabAddIns";
            this.tab1.Name = "tab1";
            // 
            // AsymptoteGroup
            // 
            this.AsymptoteGroup.Items.Add(this.btnAddAsy);
            this.AsymptoteGroup.Items.Add(this.btnLoadAsyCode);
            this.AsymptoteGroup.Items.Add(this.btnConfigure);
            this.AsymptoteGroup.Label = "Asymptote";
            this.AsymptoteGroup.Name = "AsymptoteGroup";
            // 
            // btnAddAsy
            // 
            this.btnAddAsy.Label = "Add Picture";
            this.btnAddAsy.Name = "btnAddAsy";
            this.btnAddAsy.ShowImage = true;
            this.btnAddAsy.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.btnAddAsy_Click);
            // 
            // btnLoadAsyCode
            // 
            this.btnLoadAsyCode.Label = "Load ASY Code";
            this.btnLoadAsyCode.Name = "btnLoadAsyCode";
            this.btnLoadAsyCode.ShowImage = true;
            this.btnLoadAsyCode.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.btnLoadAsy_Click);
            // 
            // btnConfigure
            // 
            this.btnConfigure.Label = "Configure";
            this.btnConfigure.Name = "btnConfigure";
            this.btnConfigure.ShowImage = true;
            this.btnConfigure.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.btnConfigure_Click);
            // 
            // AsyLoaderRibbon
            // 
            this.Name = "AsyLoaderRibbon";
            this.RibbonType = "Microsoft.PowerPoint.Presentation";
            this.Tabs.Add(this.tab1);
            this.Load += new Microsoft.Office.Tools.Ribbon.RibbonUIEventHandler(this.AsyLoaderRibbon_Load);
            this.tab1.ResumeLayout(false);
            this.tab1.PerformLayout();
            this.AsymptoteGroup.ResumeLayout(false);
            this.AsymptoteGroup.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        internal Microsoft.Office.Tools.Ribbon.RibbonTab tab1;
        internal Microsoft.Office.Tools.Ribbon.RibbonGroup AsymptoteGroup;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton btnAddAsy;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton btnLoadAsyCode;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton btnConfigure;
    }

    partial class ThisRibbonCollection
    {
        internal AsyLoaderRibbon AsyLoaderRibbon
        {
            get { return this.GetRibbon<AsyLoaderRibbon>(); }
        }
    }
}
