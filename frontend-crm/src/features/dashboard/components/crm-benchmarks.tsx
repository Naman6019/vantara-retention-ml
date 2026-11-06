import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ShieldCheck, Sparkles, Network } from 'lucide-react'

export function CrmBenchmarks() {
  return (
    <div className='grid gap-4 md:grid-cols-3'>
      {/* Churn */}
      <Card className='border-l-4 border-l-indigo-600 border-border/60 shadow-sm'>
        <CardHeader className='pb-2'>
          <div className='flex items-center justify-between'>
            <CardTitle className='text-sm font-semibold flex items-center gap-2'>
              <ShieldCheck className='size-4 text-indigo-600' />
              Churn Classifier (XGBoost)
            </CardTitle>
            <Badge variant='outline' className='bg-indigo-50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-300'>
              Production Core
            </Badge>
          </div>
        </CardHeader>
        <CardContent className='text-xs space-y-1.5 text-muted-foreground'>
          <div className='flex justify-between py-0.5 border-b border-border/40'>
            <span>Test ROC-AUC:</span>
            <span className='font-semibold text-foreground font-mono'>0.8080</span>
          </div>
          <div className='flex justify-between py-0.5 border-b border-border/40'>
            <span>Recall at 90-Day Horizon:</span>
            <span className='font-semibold text-foreground font-mono'>81.3%</span>
          </div>
          <div className='flex justify-between py-0.5'>
            <span>Cross-Validation:</span>
            <span className='font-medium text-foreground'>5-Fold Stratified CV</span>
          </div>
        </CardContent>
      </Card>

      {/* CLV */}
      <Card className='border-l-4 border-l-emerald-600 border-border/60 shadow-sm'>
        <CardHeader className='pb-2'>
          <div className='flex items-center justify-between'>
            <CardTitle className='text-sm font-semibold flex items-center gap-2'>
              <Sparkles className='size-4 text-emerald-600' />
              CLV Regressor (Ridge)
            </CardTitle>
            <Badge variant='outline' className='bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300'>
              Calibrated
            </Badge>
          </div>
        </CardHeader>
        <CardContent className='text-xs space-y-1.5 text-muted-foreground'>
          <div className='flex justify-between py-0.5 border-b border-border/40'>
            <span>Test R² Score:</span>
            <span className='font-semibold text-foreground font-mono'>0.8447</span>
          </div>
          <div className='flex justify-between py-0.5 border-b border-border/40'>
            <span>Mean Absolute Error:</span>
            <span className='font-semibold text-foreground font-mono'>£370.22</span>
          </div>
          <div className='flex justify-between py-0.5'>
            <span>Horizon:</span>
            <span className='font-medium text-foreground'>Forward 90-Day Net Spend</span>
          </div>
        </CardContent>
      </Card>

      {/* Segmentation */}
      <Card className='border-l-4 border-l-amber-500 border-border/60 shadow-sm'>
        <CardHeader className='pb-2'>
          <div className='flex items-center justify-between'>
            <CardTitle className='text-sm font-semibold flex items-center gap-2'>
              <Network className='size-4 text-amber-500' />
              Segmentation Engine (K-Means)
            </CardTitle>
            <Badge variant='outline' className='bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'>
              k = 4 Clusters
            </Badge>
          </div>
        </CardHeader>
        <CardContent className='text-xs space-y-1.5 text-muted-foreground'>
          <div className='flex justify-between py-0.5 border-b border-border/40'>
            <span>Silhouette Score:</span>
            <span className='font-semibold text-foreground font-mono'>0.4612</span>
          </div>
          <div className='flex justify-between py-0.5 border-b border-border/40'>
            <span>Personas:</span>
            <span className='font-semibold text-foreground'>Champions, Loyal, At-Risk, Dormant</span>
          </div>
          <div className='flex justify-between py-0.5'>
            <span>Latent Space:</span>
            <span className='font-medium text-foreground'>RFM + Return + Engagement</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
