import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Users, AlertTriangle, TrendingUp, CalendarClock } from 'lucide-react'
import crmData from '@/data/crm-data.json'

export function CrmKpiCards() {
  const { kpis } = crmData

  return (
    <div className='grid gap-4 sm:grid-cols-2 lg:grid-cols-4'>
      {/* 1. Accounts */}
      <Card className='border-border/60 bg-card shadow-sm hover:shadow-md transition-all'>
        <CardHeader className='flex flex-row items-center justify-between pb-2'>
          <CardTitle className='text-xs font-semibold uppercase tracking-wider text-muted-foreground'>
            Total CRM Accounts
          </CardTitle>
          <div className='flex size-8 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600 dark:bg-indigo-950/50 dark:text-indigo-400'>
            <Users className='size-4' />
          </div>
        </CardHeader>
        <CardContent>
          <div className='text-2xl font-bold tracking-tight text-foreground'>
            {kpis.total_accounts.toLocaleString()}
          </div>
          <div className='mt-2 flex items-center gap-1.5'>
            <Badge variant='outline' className='border-indigo-200 bg-indigo-50/50 text-indigo-700 dark:border-indigo-900/50 dark:bg-indigo-950/40 dark:text-indigo-300'>
              100% Data Lake Synchronized
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* 2. Churn Risk */}
      <Card className='border-border/60 bg-card shadow-sm hover:shadow-md transition-all'>
        <CardHeader className='flex flex-row items-center justify-between pb-2'>
          <CardTitle className='text-xs font-semibold uppercase tracking-wider text-muted-foreground'>
            Portfolio Churn Risk
          </CardTitle>
          <div className='flex size-8 items-center justify-center rounded-lg bg-rose-50 text-rose-600 dark:bg-rose-950/50 dark:text-rose-400'>
            <AlertTriangle className='size-4' />
          </div>
        </CardHeader>
        <CardContent>
          <div className='text-2xl font-bold tracking-tight text-rose-600 dark:text-rose-400'>
            {kpis.churn_rate.toFixed(1)}%
          </div>
          <div className='mt-2 flex items-center gap-1.5'>
            <Badge variant='destructive' className='bg-rose-100 text-rose-700 hover:bg-rose-100 dark:bg-rose-950/60 dark:text-rose-300'>
              Target: &lt; 45.0% | Active Risk
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* 3. ARPA */}
      <Card className='border-border/60 bg-card shadow-sm hover:shadow-md transition-all'>
        <CardHeader className='flex flex-row items-center justify-between pb-2'>
          <CardTitle className='text-xs font-semibold uppercase tracking-wider text-muted-foreground'>
            Avg Revenue Per Account (ARPA)
          </CardTitle>
          <div className='flex size-8 items-center justify-center rounded-lg bg-emerald-50 text-emerald-600 dark:bg-emerald-950/50 dark:text-emerald-400'>
            <TrendingUp className='size-4' />
          </div>
        </CardHeader>
        <CardContent>
          <div className='text-2xl font-bold tracking-tight text-foreground'>
            £{kpis.arpa.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className='mt-2 flex items-center gap-1.5'>
            <Badge variant='outline' className='border-emerald-200 bg-emerald-50/50 text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300'>
              +8.4% YoY Expansion
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* 4. Recency */}
      <Card className='border-border/60 bg-card shadow-sm hover:shadow-md transition-all'>
        <CardHeader className='flex flex-row items-center justify-between pb-2'>
          <CardTitle className='text-xs font-semibold uppercase tracking-wider text-muted-foreground'>
            Avg Account Recency
          </CardTitle>
          <div className='flex size-8 items-center justify-center rounded-lg bg-amber-50 text-amber-600 dark:bg-amber-950/50 dark:text-amber-400'>
            <CalendarClock className='size-4' />
          </div>
        </CardHeader>
        <CardContent>
          <div className='text-2xl font-bold tracking-tight text-foreground'>
            {kpis.avg_recency.toFixed(1)} Days
          </div>
          <div className='mt-2 flex items-center gap-1.5'>
            <Badge variant='outline' className='border-amber-200 bg-amber-50/50 text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-300'>
              Intervention Window: 90 Days
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
