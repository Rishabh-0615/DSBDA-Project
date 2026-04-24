document.addEventListener("DOMContentLoaded", function() {
    const chartInstances = {};

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    font: { family: "'Inter', sans-serif" }
                }
            }
        },
        animation: {
            duration: 1500,
            easing: 'easeOutQuart'
        }
    };

    const buildUrl = (base, params) => {
        const searchParams = new URLSearchParams();
        Object.keys(params || {}).forEach((key) => {
            const value = params[key];
            if (value !== null && value !== undefined && value !== '') {
                searchParams.set(key, value);
            }
        });

        const query = searchParams.toString();
        return query ? `${base}?${query}` : base;
    };

    const renderOrUpdate = (key, canvasId, config) => {
        const canvas = document.getElementById(canvasId);
        if(!canvas) return;

        if(chartInstances[key]) {
            chartInstances[key].destroy();
        }
        chartInstances[key] = new Chart(canvas.getContext('2d'), config);
    };

    const analysisFilters = () => {
        const region = document.getElementById('analysisRegionFilter')?.value || 'all';
        const category = document.getElementById('analysisCategoryFilter')?.value || 'all';
        const startMonth = document.getElementById('analysisStartMonth')?.value || '';
        const endMonth = document.getElementById('analysisEndMonth')?.value || '';

        return {
            region: region,
            category: category,
            start_month: startMonth,
            end_month: endMonth
        };
    };

    // Only fetch if we're on a page with charts (dashboard or analysis)
    if(document.getElementById('monthlyTrendChart') || document.getElementById('analysisMonthlyTrend')) {
        const loadCharts = (params = null) => {
            const apiUrl = buildUrl('/api/chart-data', params);

            fetch(apiUrl)
            .then(res => res.json())
            .then(data => {
                if(data.error) return;

                // Dashboard Charts
                if(document.getElementById('monthlyTrendChart')) {
                    renderOrUpdate('monthlyTrendChart', 'monthlyTrendChart', {
                        type: 'line',
                        data: {
                            labels: data.monthly_trend.labels,
                            datasets: [{
                                label: 'Monthly Sales (₹)',
                                data: data.monthly_trend.values,
                                borderColor: '#0d47a1',
                                backgroundColor: 'rgba(13, 71, 161, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.4
                            }]
                        },
                        options: commonOptions
                    });

                    renderOrUpdate('categorySalesChart', 'categorySalesChart', {
                        type: 'doughnut',
                        data: {
                            labels: data.category_sales.labels,
                            datasets: [{
                                data: data.category_sales.values,
                                backgroundColor: ['#0d47a1', '#00d2ff', '#1a237e', '#82b1ff'],
                                borderWidth: 0
                            }]
                        },
                        options: { ...commonOptions, plugins: { legend: { position: 'right' } } }
                    });

                    renderOrUpdate('regionSalesChart', 'regionSalesChart', {
                        type: 'bar',
                        data: {
                            labels: data.region_sales.labels,
                            datasets: [{
                                label: 'Sales by Region (₹)',
                                data: data.region_sales.values,
                                backgroundColor: '#00d2ff',
                                borderRadius: 5
                            }]
                        },
                        options: { ...commonOptions, scales: { y: { beginAtZero: true } } }
                    });

                    renderOrUpdate('segmentProfitChart', 'segmentProfitChart', {
                        type: 'bar',
                        data: {
                            labels: data.segment_profit.labels,
                            datasets: [{
                                label: 'Profit by Segment (₹)',
                                data: data.segment_profit.values,
                                backgroundColor: '#1a237e',
                                borderRadius: 5
                            }]
                        },
                        options: { ...commonOptions, scales: { y: { beginAtZero: true } } }
                    });
                }

                // Analysis Charts
                if(document.getElementById('analysisMonthlyTrend')) {
                    renderOrUpdate('analysisMonthlyTrend', 'analysisMonthlyTrend', {
                        type: 'line',
                        data: {
                            labels: data.monthly_trend.labels,
                            datasets: [{
                                label: 'Sales Trend (₹)',
                                data: data.monthly_trend.values,
                                borderColor: '#00d2ff',
                                backgroundColor: 'rgba(0, 210, 255, 0.2)',
                                borderWidth: 3,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 4,
                                pointHoverRadius: 6
                            }]
                        },
                        options: commonOptions
                    });

                    renderOrUpdate('analysisRegionChart', 'analysisRegionChart', {
                        type: 'pie',
                        data: {
                            labels: data.region_sales.labels,
                            datasets: [{
                                data: data.region_sales.values,
                                backgroundColor: ['#0d47a1', '#00d2ff', '#1a237e', '#82b1ff'],
                                borderWidth: 2,
                                borderColor: '#ffffff'
                            }]
                        },
                        options: commonOptions
                    });

                    renderOrUpdate('analysisCategoryChart', 'analysisCategoryChart', {
                        type: 'bar',
                        data: {
                            labels: data.category_sales.labels,
                            datasets: [{
                                label: 'Revenue (₹)',
                                data: data.category_sales.values,
                                backgroundColor: '#1a237e',
                                borderRadius: 8
                            }]
                        },
                        options: { ...commonOptions, scales: { y: { beginAtZero: true } } }
                    });
                }
            })
            .catch(err => console.error("Error loading chart data:", err));

        };

        loadCharts();

        const applyButton = document.getElementById('analysisApplyFilters');
        if(applyButton) {
            applyButton.addEventListener('click', function() {
                loadCharts(analysisFilters());
            });
        }
    }
});
