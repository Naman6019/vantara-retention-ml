import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { TopNav } from '@/components/layout/top-nav'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'

import { CrmKpiCards } from './components/crm-kpi-cards'
import { CrmRevenueChart } from './components/crm-revenue-chart'
import { CrmBenchmarks } from './components/crm-benchmarks'
import { CrmSegmentation } from './components/crm-segmentation'
import { CrmLeaderboard } from './components/crm-leaderboard'
import { CrmCustomer360 } from './components/crm-customer-360'
import { CrmBatch } from './components/crm-batch'

export function Dashboard() {
  return (
    <>
      {/* ===== Top App Header ===== */}
      <Header>
        <TopNav links={topNav} className='me-auto' />
        <div className='flex items-center gap-2'>
          <Badge variant='outline' className='hidden sm:inline-flex items-center gap-1.5 border-emerald-300 bg-emerald-50/50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300 text-xs py-1'>
            <span className='size-2 rounded-full bg-emerald-500 animate-pulse' />
            Production Core v2.4
          </Badge>
          <Search />
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      {/* ===== Main Content Area ===== */}
      <Main>
        {/* Title & Metadata Strip */}
        <div className='mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between'>
          <div>
            <h1 className='text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl'>
              🏢 Vantara Retail — Enterprise CRM
            </h1>
            <p className='text-xs sm:text-sm text-muted-foreground mt-1'>
              Customer Relationship Management (CRM) & Predictive Retention Intelligence Platform
            </p>
          </div>

          <div className='flex flex-wrap items-center gap-2'>
            <Badge variant='outline' className='bg-indigo-50/50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-300 font-mono text-xs'>
              5,225 Accounts Synced
            </Badge>
            <Badge variant='secondary' className='text-xs'>
              Role: Retention Ops & CS
            </Badge>
          </div>
        </div>

        {/* CRM Tabs System */}
        <Tabs defaultValue='overview' className='space-y-4'>
          <div className='w-full overflow-x-auto pb-1 border-b border-border/60'>
            <TabsList className='bg-muted/60 p-1'>
              <TabsTrigger value='overview' className='text-xs font-semibold'>
                📊 Executive Cockpit
              </TabsTrigger>
              <TabsTrigger value='segmentation' className='text-xs font-semibold'>
                🎯 Segmentation & Personas
              </TabsTrigger>
              <TabsTrigger value='leaderboard' className='text-xs font-semibold'>
                ⚠️ Retention Leaderboard
              </TabsTrigger>
              <TabsTrigger value='customer360' className='text-xs font-semibold'>
                🔍 Customer 360 & XAI
              </TabsTrigger>
              <TabsTrigger value='batch' className='text-xs font-semibold'>
                📁 Batch Scoring & Sync
              </TabsTrigger>
            </TabsList>
          </div>

          {/* 1. Executive Cockpit */}
          <TabsContent value='overview' className='space-y-6'>
            <CrmKpiCards />
            <CrmRevenueChart />
            <div>
              <h2 className='text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3'>
                Production Model Service Benchmarks
              </h2>
              <CrmBenchmarks />
            </div>
          </TabsContent>

          {/* 2. Customer Segmentation */}
          <TabsContent value='segmentation' className='space-y-4'>
            <CrmSegmentation />
          </TabsContent>

          {/* 3. Retention Leaderboard */}
          <TabsContent value='leaderboard' className='space-y-4'>
            <CrmLeaderboard />
          </TabsContent>

          {/* 4. Customer 360 & XAI */}
          <TabsContent value='customer360' className='space-y-4'>
            <CrmCustomer360 />
          </TabsContent>

          {/* 5. Batch Scoring & CRM Sync */}
          <TabsContent value='batch' className='space-y-4'>
            <CrmBatch />
          </TabsContent>
        </Tabs>
      </Main>
    </>
  )
}

const topNav = [
  {
    title: 'CRM Intelligence',
    href: '/',
    isActive: true,
    disabled: false,
  },
  {
    title: 'Retention Tasks',
    href: '/tasks',
    isActive: false,
    disabled: false,
  },
  {
    title: 'Account Directory',
    href: '/users',
    isActive: false,
    disabled: false,
  },
]
